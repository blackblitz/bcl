"""Autodiff Hessian."""

import jax.numpy as jnp
from jax import flatten_util, grad, jacfwd, jit, random, tree_util

from . import make_loss_reg, make_step


def make_loss_ah(state, loss):
    return jit(
        lambda params, x, y:
        0.5 * (d := flatten_util.ravel_pytree(params)[0] - state.hyperparams['minimum'])
        @ state.hyperparams['hessian'] @ d
        + loss(params, x, y)
    )


def ah(nepochs, state, loss_basic, dataset, precision=0.1, diagonal=False):
    state = state.replace(hyperparams={'precision': precision})
    loss = make_loss_reg(state, loss_basic)
    step = make_step(loss)
    for i, (x, y) in enumerate(dataset.train()):
        for _ in range(nepochs):
            state = step(state, x, y)
        yield loss, state, x, y
        pflat, punflatten = flatten_util.ravel_pytree(state.params)
        hessian = jacfwd(grad(lambda p: loss_basic(punflatten(p), x, y)))(pflat)
        state = state.replace(hyperparams={
            'minimum': pflat,
            'hessian': (
                state.hyperparams.get(
                    'hessian',
                    jnp.diag(jnp.full_like(pflat, precision))
                ) + (jnp.diag(jnp.diag(hessian)) if diagonal else hessian)
            )
        })
        loss = make_loss_ah(state, loss_basic)
        step = make_step(loss)
