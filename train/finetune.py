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
            kwargs = {'batch_size': len(dataset), 'collate_fn': numpy_collate}
        else:
            kwargs = dataloader_kwargs
        for _ in range(nepochs):
            for x, y in DataLoader(dataset, **kwargs):
                state = step(state, x, y)
        yield state, loss