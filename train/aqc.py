"""Autodiff Quadratic Consolidation."""

import jax.numpy as jnp
from jax import flatten_util, grad, jacfwd, jit

from dataio import iter_batches

from .loss import make_loss_reg

from . import Trainer


class AutodiffQuadraticConsolidation(Trainer):
    def update_loss(self):
        if hasattr(self, 'loss'):
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
                self.state, self.hyperparams['precision'], self.loss_basic
            )


    def update_hyperparams(self, batch_size, x, y):
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
                    lambda p: self.loss_basic(punflatten(p), x_batch, y_batch)
                ))(pflat)
                for x_batch, y_batch in iter_batches(1, batch_size, x, y)
            )
        )
