"""Autodiff Hessian."""

import jax.numpy as jnp
from jax import flatten_util, grad, jacfwd, jit


def make_loss(state, hyperparams, loss_basic):
    return jit(
        lambda params, x, y:
        0.5 * (d := flatten_util.ravel_pytree(params)[0] - hyperparams['minimum'])
        @ hyperparams['hessian'] @ d
        + loss_basic(params, x, y)
    )


def update_hyperparams(state, hyperparams, loss_basic, batches):
    pflat, punflatten = flatten_util.ravel_pytree(state.params)
    hyperparams['minimum'] = pflat
    hyperparams['hessian'] = (
        hyperparams.get(
            'hessian',
            jnp.diag(
                jnp.full_like(
                    flatten_util.ravel_pytree(state.params)[0],
                    hyperparams['precision']
                )
            )
        ) + sum(
            jacfwd(grad(lambda p: loss_basic(punflatten(p), x, y)))(pflat)
            for x, y in batches
        )
    )
