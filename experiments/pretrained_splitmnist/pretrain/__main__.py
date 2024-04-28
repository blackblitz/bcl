"""Script to pre-train for Split MNIST."""

from importlib.resources import files

from flax.training import orbax_utils
from jax.nn import softmax
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer
from torchvision.datasets import EMNIST
from tqdm import tqdm

from .models import cnnswish, cnntanh, make_state_pretrained
from evaluate.softmax import accuracy
from torchds import fetch
from train import make_loss_sce, make_step


def rmtree(path):
    if path.exists():
        for p in path.iterdir():
            if p.is_file():
                p.unlink()
            else:
                rmtree(p)
        path.rmdir()


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
    loss = make_loss_sce(state)
    step = make_step(loss)
    for i, (x, y) in tqdm(enumerate(fetch(emnist_train, 3,  64))):
        state = step(state, x, y)
    print(accuracy(state, fetch(emnist_test, 1, 1024)))
    path = files('experiments.pretrained_splitmnist.pretrain') / name
    rmtree(path)
    PyTreeCheckpointer().save(
        path,
        state.params,
        save_args=orbax_utils.save_args_from_target(state.params)
    )
