"""Autodiff Hessian."""

import jax.numpy as jnp
from jax import flatten_util, grad, jacfwd, jit, tree_util

from datasets import fetch
from . import make_loss_reg, make_step


def make_loss_ah(state, hyperparams, loss_basic):
    return jit(
        lambda params, x, y:
        0.5 * (d := flatten_util.ravel_pytree(params)[0] - hyperparams['minimum'])
        @ hyperparams['hessian'] @ d
        + loss_basic(params, x, y)
    )


def ah(
    make_loss_basic, num_epochs, batch_size, state, hyperparams, dataset
):
    if hyperparams['init']:
        make_loss = make_loss_reg
        hyperparams['hessian'] = jnp.diag(
            jnp.full_like(
                flatten_util.ravel_pytree(state.params)[0],
                hyperparams['precision']
            )
        )
    else:
        make_loss = make_loss_ah
    loss_basic = make_loss_basic(state)
    loss = make_loss(state, hyperparams, loss_basic)
    step = make_step(loss)
    for x, y in fetch(dataset, num_epochs, batch_size):
        state = step(state, x, y)
    pflat, punflatten = flatten_util.ravel_pytree(state.params)
    hyperparams['minimum'] = pflat
    hyperparams['hessian'] = (
        hyperparams['hessian']
        + sum(
            jacfwd(grad(lambda p: loss_basic(punflatten(p), x, y)))(pflat)
            for x, y in fetch(dataset, 1, batch_size)
        )
    )
    hyperparams['init'] = False
    return state, hyperparams, loss