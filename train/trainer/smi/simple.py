"""Simple trainers."""

from operator import add, mul, sub

from flax.training.train_state import TrainState
from jax import flatten_util, grad, jacfwd, jit, nn, random, tree_util, vmap
import jax.numpy as jnp
from optax import adam, constant_schedule, cosine_onecycle_schedule

from dataops import tree
from dataops.array import batch, shuffle
from models import NLL
from models.fcnn import FCNN3

from . import MAPMixin
from .. import OptimizingTrainer
from ..coreset import JointCoreset
from ...training import init, make_step
from ...training.loss import (
    diag_quad_con, flat_quad_con, huber, l2_reg, neu_con
)


class Joint(MAPMixin, OptimizingTrainer):
    """Joint training."""

    def __init__(self, model, mspec, hparams):
        """Initialize self."""
        super().__init__(model, mspec, hparams)
        self.hparams['coreset'] = JointCoreset(
            'coreset.zarr', 'coreset.memmap', self.mspec
        )

    def update_loss(self, xs, ys):
        """Update the loss function."""
        self.loss = jit(
            l2_reg(self.hparams['precision'], self.hparams['nll'])
        )

    def update_state(self, xs, ys):
        """Update the training state."""
        key1, key2, key3 = random.split(
            self.hparams['keys']['update_state'], num=3
        )
        self.hparams['coreset'].update(key1, xs, ys)
        self.hparams['coreset'].create_memmap()
        step = make_step(self.loss)
        self.state = self._init_state(key2, len(ys))
        for key in random.split(key3, num=self.hparams['n_epochs']):
            for xs_batch, ys_batch in (
                self.hparams['coreset'].shuffle_batch(
                    key, self.hparams['batch_size']
                )
            ):
                self.state = step(self.state, xs_batch, ys_batch)
            yield self.state
        self.hparams['coreset'].delete_memmap()

    def update_hparams(self, xs, ys):
        """Update mutables."""


class RegularTrainer(MAPMixin, OptimizingTrainer):
    """Mixin for regular SGD."""

    def update_state(self, xs, ys):
        """Update the training state."""
        key1, key2 = random.split(self.hparams['keys']['update_state'])
        step = make_step(self.loss)
        self.state = self._init_state(key1, len(ys))
        for key in random.split(key2, num=self.hparams['n_epochs']):
            for indices in batch(
                self.hparams['batch_size'], shuffle(key, len(ys))
            ):
                self.state = step(self.state, xs[indices], ys[indices])
            yield self.state


class Finetuning(RegularTrainer):
    """Fine-tuning for continual learning."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        self.loss = jit(
            l2_reg(self.hparams['precision'], self.hparams['nll'])
        )

    def update_hparams(self, xs, ys):
        """Update the mutable hyperparameters."""


class QuadraticConsolidation(RegularTrainer):
    """Quadratic consolidiation."""

    def __init__(self, model, mspec, hparams):
        """Initialize self."""
        super().__init__(model, mspec, hparams)
        self.hparams |= {
            'minimum': tree.full_like(self.hparams['param_example'], 0.0),
            'hessian': tree.full_like(
                self.hparams['param_example'], self.hparams['precision']
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        if self.loss is None:
            self.loss = jit(
                l2_reg(self.hparams['precision'], self.hparams['nll'])
            )
        else:
            self.loss = jit(
                diag_quad_con(
                    self.hparams['lambda'],
                    self.hparams['minimum'],
                    self.hparams['hessian'],
                    self.hparams['nll']
                )
            )


class ElasticWeightConsolidation(QuadraticConsolidation):
    """Elastic weight consolidation."""

    def update_hparams(self, xs, ys):
        """Update the mutable hyperparameters."""
        self.hparams['minimum'] = self.state.params
        nll_grad = grad(self.hparams['nll'])

        @jit
        def mean_squared_grad(x):
            xs = jnp.expand_dims(x, 0)
            out = self.model.apply({'params': self.state.params}, xs)[0]
            match self.mspec.nll:
                case NLL.SIGMOID_CROSS_ENTROPY:
                    pred = nn.softmax(jnp.array([0., out[0]]))
                case NLL.SOFTMAX_CROSS_ENTROPY:
                    pred = nn.softmax(out)
            return tree_util.tree_map(
                lambda g: jnp.tensordot(pred, g ** 2, axes=(0, 0)),
                vmap(nll_grad, in_axes=(None, None, 0))(
                    self.state.params, xs, jnp.arange(len(pred))[:, None]
                )
            )

        fisher = tree_util.tree_map(jnp.zeros_like, self.state.params)
        for x in xs:
            fisher = tree_util.tree_map(add, fisher, mean_squared_grad(x))
        fisher = tree_util.tree_map(lambda x: x / len(ys), fisher)
        self.hparams['hessian'] = tree_util.tree_map(
            add, self.hparams['hessian'], fisher
        )


class SynapticIntelligence(QuadraticConsolidation):
    """Synaptic Intelligence."""

    def update_state(self, xs, ys):
        """Update the training state."""
        key1, key2 = random.split(
            self.hparams['keys']['update_state']
        )

        @jit
        def step(state, loss_change, xs, ys):
            grads = grad(self.loss)(state.params, xs, ys)
            next_state = state.apply_gradients(grads=grads)
            diff = tree_util.tree_map(sub, next_state.params, state.params)
            loss_change = tree_util.tree_map(
                add, loss_change, tree_util.tree_map(mul, grads, diff)
            )
            return next_state, loss_change

        self.state = self._init_state(key1, len(ys))
        loss_change = tree.full_like(self.state.params, 0.0)
        start_params = self.state.params
        for key in random.split(key2, num=self.hparams['n_epochs']):
            for indices in batch(
                self.hparams['batch_size'], shuffle(key, len(ys))
            ):
                self.state, loss_change = step(
                    self.state, loss_change, xs[indices], ys[indices]
                )
            yield self.state

        self.hparams['minimum'] = self.state.params
        self.hparams['hessian'] = tree_util.tree_map(
            add,
            self.hparams['hessian'],
            tree_util.tree_map(
                lambda dl, dp: dl / (self.hparams['xi'] + dp ** 2),
                loss_change,
                tree_util.tree_map(sub, self.state.params, start_params)
            )
        )

    def update_hparams(self, xs, ys):
        """Update the mutable hyperparameters."""


class AutodiffQuadraticConsolidation(RegularTrainer):
    """Autodiff Quadratic Consolidation."""

    def __init__(self, model, mspec, hparams):
        """Initialize self."""
        super().__init__(model, mspec, hparams)
        flat_params = flatten_util.ravel_pytree(
            self.hparams['param_example']
        )[0]
        self.hparams |= {
            'flat_minimum': jnp.zeros_like(flat_params),
            'flat_hessian': jnp.diag(
                jnp.full_like(flat_params, self.hparams['precision'])
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        if self.loss is None:
            self.loss = jit(
                l2_reg(self.hparams['precision'], self.hparams['nll'])
            )
        else:
            self.loss = jit(
                flat_quad_con(
                    self.hparams['lambda'],
                    self.hparams['flat_minimum'],
                    self.hparams['flat_hessian'],
                    self.hparams['nll']
                )
            )

    def update_hparams(self, xs, ys):
        """Update hyperparameters."""
        flat_params, unflatten = flatten_util.ravel_pytree(self.state.params)
        self.hparams['flat_minimum'] = flat_params

        def flat_nll(flat_params, xs, ys):
            return self.hparams['nll'](unflatten(flat_params), xs, ys)

        flat_nll_hessian = jit(jacfwd(grad(flat_nll)))
        self.hparams['flat_hessian'] = self.hparams['flat_hessian'] + sum(
            flat_nll_hessian(flat_params, xs[indices], ys[indices])
            for indices in batch(
                self.hparams['pass_size'], jnp.arange(len(ys))
            )
        )


class NeuralConsolidation(RegularTrainer):
    """Neural Consolidation."""

    def __init__(self, model, mspec, hparams):
        """Initialize self."""
        super().__init__(model, mspec, hparams)
        self.hparams |= {
            'minimum': tree.full_like(self.hparams['param_example'], 0.0),
            'con_state': None
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        if self.loss is None:
            self.loss = jit(
                l2_reg(self.hparams['precision'], self.hparams['nll'])
            )
        else:
            self.loss = jit(
                neu_con(self.hparams['con_state'], self.hparams['nll'])
            )

    def _make_con_lr_schedule(self):
        """Return the consolidator learning rate schedule."""
        match self.hparams['con_lr_schedule']:
            case 'constant':
                return constant_schedule(self.hparams['con_base_lr'])
            case 'onecycle':
                return cosine_onecycle_schedule(
                    transition_steps=(self.hparams['con_n_steps']),
                    peak_value=self.hparams['con_base_lr']
                )

    def _make_con_data(self, key, xs, ys):
        """Generate data for training the consolidator."""
        flat_params, unflatten = flatten_util.ravel_pytree(
            self.hparams['minimum']
        )
        flat_params = (
            flat_params + self.hparams['con_radius'] * random.ball(
                key, len(flat_params),
                shape=(self.hparams['con_sample_size'],)
            )
        )
        params = vmap(unflatten)(flat_params)
        loss_vals = sum(
            vmap(
                self.loss, in_axes=(0, None, None)
            )(params, xs[indices], ys[indices])
            for indices in batch(
                self.hparams['pass_size'], jnp.arange(len(ys))
            )
        )
        return flat_params, loss_vals

    def update_hparams(self, xs, ys):
        """Update hyperparameters."""
        self.hparams['minimum'] = self.state.params
        flat_params, unflatten = flatten_util.ravel_pytree(
            self.hparams['param_example']
        )
        model = FCNN3(
            dense0=self.hparams['con_dense0'],
            dense1=self.hparams['con_dense1'],
            dense2=1
        )
        loss = l2_reg(
            self.hparams['con_precision'],
            huber(model.apply)
        )
        step = make_step(loss)
        state = TrainState.create(
            apply_fn=model.apply,
            params=init(
                self.hparams['keys']['update_hparams'],
                model, flat_params.shape
            ) if self.hparams['con_state'] is None
            else self.hparams['con_state'].params,
            tx=adam(self._make_con_lr_schedule())
        )
        for key in random.split(
            self.hparams['keys']['update_hparams'],
            num=self.hparams['con_n_steps']
        ):
            flat_params, loss_values = self._make_con_data(key, xs, ys)
            state = step(state, flat_params, loss_values)
        self.hparams['con_state'] = state
