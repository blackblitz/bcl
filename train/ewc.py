"""Elastic Weight Consolidation."""

from operator import add

import jax.numpy as jnp
from jax import grad, jit, random, tree_util
from jax.lax import scan
from torch.utils.data import DataLoader

from dataseqs import numpy_collate
from . import make_loss_reg, make_step


def make_loss_ewc(state, loss):
    return jit(
        lambda params, x, y:
        0.5 * state.hyperparams['lambda'] * tree_util.tree_reduce(
            add,
            tree_util.tree_map(
                lambda hess, params, p_min: (hess * (params - p_min) ** 2).sum(),
                state.hyperparams['fisher'], params, state.hyperparams['minimum']
            )
        ) + loss(params, x, y)
    )


def get_fisher(loss, params, dataset):
    zero = tree_util.tree_map(jnp.zeros_like, params)
    total = zero
    for x, y in DataLoader(dataset, batch_size=64, collate_fn=numpy_collate):
        total_batch = scan(
            lambda c, a: (
                tree_util.tree_map(
                    lambda c, a: c + a ** 2, c,
                    grad(loss)(
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
        total = tree_util.tree_map(lambda x, y: x + y, total, total_batch)
    return tree_util.tree_map(lambda x: x / len(dataset), total)


def ewc(
    nepochs, state, loss_basic, dataseq,
    precision=0.1, lambda_=1.0, dataloader_kwargs=None
):
    state = state.replace(hyperparams={'precision': precision})
    loss = make_loss_reg(state, loss_basic)
    step = make_step(loss)
    for i, dataset in enumerate(dataseq.train()):
        if dataloader_kwargs is None:
            x, y = next(iter(
                DataLoader(
                    dataset,
                    batch_size=len(dataset),
                    collate_fn=numpy_collate
                )
            ))
            for _ in range(nepochs):
                state = step(state, x, y)
        else:
            for _ in range(nepochs):
                for x, y in DataLoader(dataset, **dataloader_kwargs):
                    state = step(state, x, y)
        yield state, loss
        state = state.replace(hyperparams={
            'lambda': lambda_,
            'minimum': state.params,
            'fisher': tree_util.tree_map(
                add,
                state.hyperparams.get(
                    'fisher', tree_util.tree_map(
                        lambda x: jnp.full_like(x, precision), state.params
                    )
                ),
                get_fisher(loss_basic, state.params, dataset)
            )
        })
        loss = make_loss_ewc(state, loss_basic)
        step = make_step(loss)