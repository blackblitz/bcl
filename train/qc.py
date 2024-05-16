"""Quadratic consolidation."""

from operator import add, truediv, mul, sub

import jax.numpy as jnp
from jax import flatten_util, grad, jacfwd, jit, tree_util, vmap
from jax.nn import softmax

from dataio import iter_batches

from .loss import make_loss_reg

from . import Trainer


class QuadraticConsolidation(Trainer):
    """Quadratic Consolidation."""

    def update_loss(self, x, y):
        """Update loss function."""
        if self.n_obs > 0:
            self.loss = jit(
                lambda params, x, y:
                0.5 * self.hyperparams.get('lambda', 1.0)
                * tree_util.tree_reduce(
                    add,
                    tree_util.tree_map(
                        lambda hess, params, p_min: (
                            hess * (params - p_min) ** 2
                        ).sum(),
                        self.hyperparams['hessian'],
                        params,
                        self.hyperparams['minimum']
                    )
                ) + self.loss_basic(params, x, y)
            )
        else:
            self.loss = make_loss_reg(
                self.hyperparams['precision'], self.loss_basic
            )


class ElasticWeightConsolidation(QuadraticConsolidation):
    """Elastic Weight Consolidation."""

    def _make_fisher(self, x, y):
        """Make Fisher Information Matrix."""
        def _expected_squared_grad_loss(x1):
            x1 = jnp.array([x1])
            pred = self.state.apply_fn({'params': self.state.params}, x1)[0]
            if not self.multiclass:
                pred = jnp.array([0., pred[0]])
            pred = softmax(pred)
            grad_loss = vmap(grad(self.loss_basic), in_axes=(None, None, 0))(
                self.state.params, x1, jnp.arange(len(pred))[:, None]
            )
            return tree_util.tree_map(
                lambda x: x.sum(axis=0),
                vmap(
                    lambda x, y: tree_util.tree_map(lambda a: x * a ** 2, y)
                )(pred, grad_loss)
            )

        total = tree_util.tree_map(jnp.zeros_like, self.state.params)
        for x_batch, _ in iter_batches(
            1, self.batch_size_hyperparams, x, y, shuffle=False
        ):
            total = tree_util.tree_map(
                add, total,
                tree_util.tree_map(
                    lambda x: x.sum(axis=0),
                    vmap(_expected_squared_grad_loss)(x_batch)
                )
            )
        return tree_util.tree_map(
            add,
            self.hyperparams.get(
                'hessian',
                tree_util.tree_map(
                    lambda x: jnp.full_like(x, self.hyperparams['precision']),
                    self.state.params
                )
            ),
            tree_util.tree_map(lambda x: x / len(y), total)
        )

    def update_hyperparams(self, x, y):
        """Update hyperparameters."""
        self.hyperparams['minimum'] = self.state.params
        self.hyperparams['hessian'] = self._make_fisher(x, y)


class SynapticIntelligence(QuadraticConsolidation):
    """Synaptic Intelligence."""

    def update_state(self, x, y):
        """Update state."""
        @jit
        def step(state, dloss_cum, dparams_cum, x, y):
            grads = grad(self.loss)(state.params, x, y)
            state_new = state.apply_gradients(grads=grads)
            dparams = tree_util.tree_map(sub, state_new.params, state.params)
            dloss_cum = tree_util.tree_map(
                add, dloss_cum,
                tree_util.tree_map(mul, grads, dparams)
            )
            dparams = tree_util.tree_map(add, dparams_cum, dparams)
            return dloss_cum, dparams_cum, state_new

        dloss_cum = tree_util.tree_map(jnp.zeros_like, self.state.params)
        dparams_cum = tree_util.tree_map(jnp.zeros_like, self.state.params)
        for x_batch, y_batch in iter_batches(
            self.n_epochs, self.batch_size_state, x, y
        ):
            dloss_cum, dparams_cum, self.state = step(
                self.state, dloss_cum, dparams_cum, x_batch, y_batch
            )
        self.hyperparams['minimum'] = self.state.params
        self.hyperparams['hessian'] = tree_util.tree_map(
            add,
            self.hyperparams.get(
                'hessian',
                tree_util.tree_map(
                    lambda x: jnp.full_like(x, self.hyperparams['precision']),
                    self.state.params
                )
            ),
            tree_util.tree_map(
                lambda x: x / len(y),
                tree_util.tree_map(
                    truediv, dloss_cum,
                    tree_util.tree_map(
                        lambda x: x ** 2 + self.hyperparams['xi'], dparams_cum
                    )
                )
            )
        )

    def update_hyperparams(self, x, y):
        """Update hyperparameters."""


class AutodiffQuadraticConsolidation(Trainer):
    """Autodiff Quadratic Consolidation."""

    def update_loss(self, x, y):
        """Update loss function."""
        if self.n_obs > 0:
            self.loss = jit(
                lambda params, x, y:
                0.5 * (
                    d := flatten_util.ravel_pytree(params)[0]
                    - self.hyperparams['minimum']
                ) @ self.hyperparams['hessian'] @ d
                + self.loss_basic(params, x, y)
            )
        else:
            self.loss = make_loss_reg(
                self.hyperparams['precision'], self.loss_basic
            )

    def update_hyperparams(self, x, y):
        """Update hyperparameters."""
        pflat, punflatten = flatten_util.ravel_pytree(self.state.params)
        self.hyperparams['minimum'] = pflat
        self.hyperparams['hessian'] = (
            self.hyperparams.get(
                'hessian',
                jnp.diag(
                    jnp.full_like(
                        flatten_util.ravel_pytree(self.state.params)[0],
                        self.hyperparams['precision']
                    )
                )
            ) + sum(
                jacfwd(grad(
                    lambda p, x, y: self.loss_basic(punflatten(p), x, y)
                ))(pflat, x_batch, y_batch) * (len(y_batch) / len(y))
                for x_batch, y_batch in iter_batches(
                    1, self.batch_size_hyperparams, x, y, shuffle=False
                )
            )
        )
