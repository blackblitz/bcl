"""Fine-tuning."""

from datasets import fetch
from . import make_loss_reg, make_step


def finetune(
    make_loss_basic, num_epochs, batch_size, state, hyperparams, dataset
):
    loss = make_loss_reg(state, hyperparams, make_loss_basic(state))
    step = make_step(loss)
    for x, y in fetch(dataset, num_epochs, batch_size):
        state = step(state, x, y)
    return state, hyperparams, loss