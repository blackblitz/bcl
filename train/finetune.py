"""Fine-tuning."""

from jax import jit, random, tree_util
from torch.utils.data import DataLoader

from dataseqs import numpy_collate
from . import make_loss_reg, make_step


def finetune(
    nepochs, state, loss_basic, dataseq,
    precision=0.1, dataloader_kwargs=None
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