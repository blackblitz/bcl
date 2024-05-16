"""Script to pre-train for Split CIFAR-10."""

from importlib.resources import files

from flax.training import orbax_utils
import numpy as np
from optax import softmax_cross_entropy_with_integer_labels
from orbax.checkpoint import PyTreeCheckpointer
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from .models import cnnswish, cnntanh, make_state_pretrained
from dataio import iter_batches
from dataio.datasets import memmap_dataset
from dataio.path import rmtree
from evaluate import accuracy
from train import make_step
from train.loss import make_loss, make_loss_reg

cifar100_train = CIFAR100(
    'data', download=True, train=True,
    transform=lambda x: np.asarray(x) / 255.0,
)
cifar100_test = CIFAR100(
    'data', download=True, train=False,
    transform=lambda x: np.asarray(x) / 255.0,
)
for name, model in zip(['cnnswish', 'cnntanh'], [cnnswish, cnntanh]):
    state = make_state_pretrained(model.Model())
    loss = make_loss_reg(0.1, make_loss(
        state, softmax_cross_entropy_with_integer_labels
    ))
    step = make_step(loss)
    for i, (x_batch, y_batch) in tqdm(enumerate(
        iter_batches(100, 64, *memmap_dataset(cifar100_train))
    )):
        state = step(state, x_batch, y_batch)
    x, y = memmap_dataset(cifar100_test)
    print(accuracy(True, state, 1024, *memmap_dataset(cifar100_test)))
    path = files('experiments.pretrained_splitcifar10.pretrain') / name
    rmtree(path)
    PyTreeCheckpointer().save(
        path,
        state.params,
        save_args=orbax_utils.save_args_from_target(state.params)
    )
