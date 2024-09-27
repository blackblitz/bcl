"""Simple trainers."""

from operator import add, mul, sub

from flax.training.train_state import TrainState
from jax import flatten_util, grad, jacfwd, jit, nn, random, tree_util, vmap
import jax.numpy as jnp
from optax import adam

from dataops import tree
from dataops.array import batch, shuffle
from models import NLL
from models.fcnn import FCNN3

from ..coreset import JointCoreset
from ..loss import diag_quad_con, flat_quad_con, huber, l2_reg, neu_con
from ..state.functions import init, make_step
from ..state.mixins import MAPMixin
from ..trainer import ContinualTrainer


class Joint(MAPMixin, ContinualTrainer):
    """Joint training."""

    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys(
            ['init_state', 'update_state', 'update_coreset']
        )

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': JointCoreset(
                'coreset.zarr', 'coreset.memmap', self.model_spec
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        self.loss = jit(
            l2_reg(self.immutables['precision'], self.precomputed['nll'])
        )

    def update_state(self, xs, ys):
        """Update the training state."""
        self.mutables['coreset'].update(
            self.precomputed['keys']['update_coreset'], xs, ys
        )
        self.mutables['coreset'].create_memmap()
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            key1, key2 = random.split(key)
            for xs_batch, ys_batch in (
                self.mutables['coreset'].shuffle_batch(
                    key2, self.immutables['batch_size']
                )
            ):
                self.state = step(self.state, xs_batch, ys_batch)
            yield self.state
        self.mutables['coreset'].delete_memmap()

    def update_mutables(self, xs, ys):
        """Update mutables."""


class RegularTrainer(MAPMixin, ContinualTrainer):
    """Mixin for regular SGD."""

    def update_state(self, xs, ys):
        """Update the training state."""
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            for indices in batch(
                self.immutables['batch_size'], shuffle(key, len(ys))
            ):
                self.state = step(self.state, xs[indices], ys[indices])
            yield self.state


class Finetuning(RegularTrainer):
    """Fine-tuning for continual learning."""

    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys(
            ['init_state', 'update_state']
        )

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {}

    def update_loss(self, xs, ys):
        """Update the loss function."""
        self.loss = jit(
            l2_reg(self.immutables['precision'], self.precomputed['nll'])
        )

    def update_mutables(self, xs, ys):
        """Update the mutable hyperparameters."""


class QuadraticConsolidation(RegularTrainer):
    """Quadratic consolidiation."""

    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys(
            ['init_state', 'update_state']
        )

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'minimum': tree.full_like(self.state.params, 0.0),
            'hessian': tree.full_like(
                self.state.params, self.immutables['precision']
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        if self.loss is None:
            self.loss = jit(
                l2_reg(self.immutables['precision'], self.precomputed['nll'])
            )
        else:
            self.loss = jit(
                diag_quad_con(
                    self.immutables['lambda'],
                    self.mutables['minimum'],
                    self.mutables['hessian'],
                    self.precomputed['nll']
                )
            )


class ElasticWeightConsolidation(QuadraticConsolidation):
    """Elastic weight consolidation."""

    def update_mutables(self, xs, ys):
        """Update the mutable hyperparameters."""
        self.mutables['minimum'] = self.state.params
        nll_grad = grad(self.precomputed['nll'])

        @jit
        def mean_squared_grad(x):
            xs = jnp.expand_dims(x, 0)
            out = self.model.apply({'params': self.state.params}, xs)[0]
            match self.model_spec.nll:
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
        self.mutables['hessian'] = tree_util.tree_map(
            add, self.mutables['hessian'], fisher
        )


class SynapticIntelligence(QuadraticConsolidation):
    """Synaptic Intelligence."""

    def update_state(self, xs, ys):
        """Update the training state."""

        @jit
        def step(state, loss_change, xs, ys):
            grads = grad(self.loss)(state.params, xs, ys)
            next_state = state.apply_gradients(grads=grads)
            diff = tree_util.tree_map(sub, next_state.params, state.params)
            loss_change = tree_util.tree_map(
                add, loss_change, tree_util.tree_map(mul, grads, diff)
            )
            return next_state, loss_change

        loss_change = tree.full_like(self.state.params, 0.0)
        start_params = self.state.params
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            for indices in batch(
                self.immutables['batch_size'], shuffle(key, len(ys))
            ):
                self.state, loss_change = step(
                    self.state, loss_change, xs[indices], ys[indices]
                )
            yield self.state

        self.mutables['minimum'] = self.state.params
        self.mutables['hessian'] = tree_util.tree_map(
            add,
            self.mutables['hessian'],
            tree_util.tree_map(
                lambda dl, dp: dl / (self.immutables['xi'] + dp ** 2),
                loss_change,
                tree_util.tree_map(sub, self.state.params, start_params)
            )
        )

    def update_mutables(self, xs, ys):
        """Update the mutable hyperparameters."""


class AutodiffQuadraticConsolidation(RegularTrainer):
    """Autodiff Quadratic Consolidation."""

    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys(
            ['init_state', 'update_state']
        )

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        flat_params = flatten_util.ravel_pytree(self.state.params)[0]
        return {
            'flat_minimum': jnp.zeros_like(flat_params),
            'flat_hessian': jnp.diag(
                jnp.full_like(flat_params, self.immutables['precision'])
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        if self.loss is None:
            self.loss = jit(
                l2_reg(self.immutables['precision'], self.precomputed['nll'])
            )
        else:
            self.loss = jit(
                flat_quad_con(
                    self.mutables['flat_minimum'],
                    self.mutables['flat_hessian'],
                    self.precomputed['nll']
                )
            )

    def update_mutables(self, xs, ys):
        """Update hyperparameters."""
        flat_params, unflatten = flatten_util.ravel_pytree(self.state.params)
        self.mutables['flat_minimum'] = flat_params

        def flat_nll(flat_params, xs, ys):
            return self.precomputed['nll'](unflatten(flat_params), xs, ys)

        flat_nll_hessian = jit(jacfwd(grad(flat_nll)))
        self.mutables['flat_hessian'] = self.mutables['flat_hessian'] + sum(
            flat_nll_hessian(flat_params, xs[indices], ys[indices])
            for indices in batch(
                self.precomputed['pass_size'], jnp.arange(len(ys))
            )
        )


class NeuralConsolidation(RegularTrainer):
    """Neural Consolidation."""

    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys(
            ['init_state', 'init_mutables', 'update_state', 'update_mutables']
        )

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        flat_params, unflatten = flatten_util.ravel_pytree(self.state.params)
        con_model = FCNN3(
            dense0=self.immutables['con_dense0'],
            dense1=self.immutables['con_dense1'],
            dense2=1
        )
        con_state = TrainState.create(
            apply_fn=con_model.apply,
            params=init(
                self.precomputed['keys']['init_mutables'],
                con_model, flat_params.shape
            ),
            tx=adam(self.immutables['con_lr'])
        )
        return {
            'minimum': tree.full_like(self.state.params, 0.0),
            'con_state': con_state
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        if self.loss is None:
            self.loss = jit(
                l2_reg(self.immutables['precision'], self.precomputed['nll'])
            )
        else:
            self.loss = jit(
                neu_con(self.mutables['con_state'], self.precomputed['nll'])
            )

    def _make_con_data(self, key, xs, ys):
        """Generate data for training the consolidator."""
        flat_params, unflatten = flatten_util.ravel_pytree(
            self.mutables['minimum']
        )
        flat_params = (
            flat_params + self.immutables['con_radius'] * random.ball(
                key, len(flat_params),
                shape=(self.immutables['con_sample_size'],)
            )
        )
        params = vmap(unflatten)(flat_params)
        loss_vals = sum(
            vmap(
                self.loss, in_axes=(0, None, None)
            )(params, xs[indices], ys[indices])
            for indices in batch(
                self.precomputed['pass_size'], jnp.arange(len(ys))
            )
        )
        return flat_params, loss_vals

    def update_mutables(self, xs, ys):
        """Update hyperparameters."""
        self.mutables['minimum'] = self.state.params
        state = self.mutables['con_state']
        loss = l2_reg(
            self.immutables['con_precision'],
            huber(self.mutables['con_state'].apply_fn)
        )
        step = make_step(loss)
        for key in random.split(
            self.precomputed['keys']['update_mutables'],
            num=self.immutables['con_n_steps']
        ):
            flat_params, loss_values = self._make_con_data(key, xs, ys)
            state = step(state, flat_params, loss_values)
        self.mutables['con_state'] = state
