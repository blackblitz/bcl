"""Elastic Weight Consolidation."""

from operator import add

import jax.numpy as jnp
from jax import grad, jit, tree_util, vmap
from jax.lax import scan


def make_loss(state, hyperparams, loss_basic):
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


def make_fisher(state, hyperparams, loss_basic, batches):
    total = tree_util.tree_map(jnp.zeros_like, state.params)
    count = 0
    for x, y in batches:
        total = tree_util.tree_map(
            add, total,
            tree_util.tree_map(
                lambda x: (x ** 2).sum(axis=0),
                vmap(
                    grad(loss_basic), in_axes=(None, 0, 0)
                )(state.params, jnp.expand_dims(x, 1), jnp.expand_dims(y, 1))
            )
        )
        count += len(y)
    return tree_util.tree_map(
        add,
        hyperparams.get(
            'fisher',
            tree_util.tree_map(
                lambda x: jnp.full_like(x, hyperparams['precision']), state.params
            )
        ),
        tree_util.tree_map(lambda x: x / count, total)
    )


def update_hyperparams(state, hyperparams, loss_basic, batches):
    hyperparams['minimum'] = state.params
    hyperparams['fisher'] = make_fisher(
        state, hyperparams, loss_basic, batches
    )
