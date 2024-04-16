"""Joint training."""

from torch.utils.data import ConcatDataset

from datasets import fetch
from . import make_loss_reg, make_step


def joint(
    make_loss_basic, num_epochs, batch_size, state, hyperparams, dataset
):
    loss = make_loss_reg(state, hyperparams, make_loss_basic(state))
    step = make_step(loss)
    coreset = hyperparams.get('coreset')
    concat = (
        ConcatDataset([coreset, dataset])
        if coreset is not None else dataset
    )
    for x, y in fetch(concat, num_epochs, batch_size):
        state = step(state, x, y)
    hyperparams['coreset'] = concat
    return state, hyperparams, loss
