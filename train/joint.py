"""Joint training."""

import jax.numpy as jnp
from jax import jit, random, tree_util
from torch.utils.data import ConcatDataset, DataLoader

from dataseqs import numpy_collate
from . import make_loss_reg, make_step


def joint(
    nepochs, state, loss_basic, dataseq,
    precision=0.1, dataloader_kwargs=None
):
    state = state.replace(hyperparams={'precision': precision})
    loss = make_loss_reg(state, loss_basic)
    step = make_step(loss)
    for i, dataset in enumerate(dataseq.train()):
        concat = dataset if i == 0 else ConcatDataset([concat, dataset])
        if dataloader_kwargs is None:
            x, y = next(iter(
                DataLoader(
                    concat,
                    batch_size=len(concat),
                    collate_fn=numpy_collate
                )
            ))
            for _ in range(nepochs):
                state = step(state, x, y)
        else:
            for _ in range(nepochs):
                for x, y in DataLoader(concat, **dataloader_kwargs):
                    state = step(state, x, y)
        yield state, loss, concat