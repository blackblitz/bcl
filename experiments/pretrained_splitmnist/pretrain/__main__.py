"""Script to pre-train for Split MNIST."""

from importlib.resources import files

from flax.training import orbax_utils
import numpy as np
from optax import softmax_cross_entropy_with_integer_labels
from orbax.checkpoint import PyTreeCheckpointer
from torchvision.datasets import EMNIST
from tqdm import tqdm

from dataio import iter_batches
from dataio.datasets import memmap_dataset
from dataio.path import rmtree
from evaluate import accuracy
from train import make_step
from train.loss import make_loss, make_loss_reg

from .models import cnnswish, cnntanh, make_state_pretrained

emnist_train = EMNIST(
    'data', download=True, split='letters', train=True,
    transform=lambda x: np.asarray(x)[:, :, None] / 255.0,
    target_transform=lambda x: x - 1
)
emnist_test = EMNIST(
    'data', download=True, split='letters', train=False,
    transform=lambda x: np.asarray(x)[:, :, None] / 255.0,
    target_transform=lambda x: x - 1
)
for name, model in zip(['cnnswish', 'cnntanh'], [cnnswish, cnntanh]):
    state = make_state_pretrained(model.Model())
    loss = make_loss_reg(0.1, make_loss(
        state, softmax_cross_entropy_with_integer_labels
    ))
    step = make_step(loss)
    for i, (x_batch, y_batch) in tqdm(enumerate(
        iter_batches(10, 64, *memmap_dataset(emnist_train))
    )):
        state = step(state, x_batch, y_batch)
    print(accuracy(True, state, 1024, *memmap_dataset(emnist_test)))
    path = files('experiments.pretrained_splitmnist.pretrain') / 'ckpt' / name
    rmtree(path)
    PyTreeCheckpointer().save(
        path,
        state.params,
        save_args=orbax_utils.save_args_from_target(state.params)
    )
