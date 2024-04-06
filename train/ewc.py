"""Elastic Weight Consolidation."""

from operator import add

import jax.numpy as jnp
from jax import grad, jit, random, tree_util
from jax.lax import scan


from datasets import fetch
from . import make_loss_reg, make_step


def make_loss_ewc(state, hyperparams, loss_basic):
    return jit(
        lambda params, x, y:
        0.5 * hyperparams['lambda'] * tree_util.tree_reduce(
            add,
            tree_util.tree_map(
                lambda hess, params, p_min: (hess * (params - p_min) ** 2).sum(),
                hyperparams['fisher'], params, hyperparams['minimum']
            )
        ) + loss_basic(params, x, y)
    )


def get_fisher(batch_size, loss_basic, params, dataset):
    zero = tree_util.tree_map(jnp.zeros_like, params)
    total = zero
    for x, y in fetch(dataset, 1, batch_size):
        total_batch = scan(
            lambda c, a: (
                tree_util.tree_map(
                    lambda c, a: c + a ** 2, c,
                    grad(loss_basic)(
                        params,
                        jnp.expand_dims(a['x'], 0),
                        jnp.expand_dims(a['y'], 0)
                    )
                ),
                None
            ),
            zero,
            {'x': x, 'y': y}
        )[0]
        total = tree_util.tree_map(add, total, total_batch)
    return tree_util.tree_map(lambda x: x / len(dataset), total)


def ewc(
    make_loss_basic, num_epochs, batch_size, state, hyperparams, dataset
):
    if hyperparams['init']:
        make_loss = make_loss_reg
        hyperparams['fisher'] = tree_util.tree_map(
            lambda x: jnp.full_like(x, hyperparams['precision']), state.params
        )
    else:
        make_loss = make_loss_ewc
    loss_basic = make_loss_basic(state)
    loss = make_loss(state, hyperparams, loss_basic)
    step = make_step(loss)
    for x, y in fetch(dataset, num_epochs, batch_size):
        state = step(state, x, y)
    hyperparams['minimum'] = state.params
    hyperparams['fisher'] = tree_util.tree_map(
        add, hyperparams['fisher'],
        get_fisher(batch_size, loss_basic, state.params, dataset)
    )
    hyperparams['init'] = False
    return state, hyperparams, loss