"""Script for Pre-trained Split MNIST."""

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from tqdm import tqdm

from train import make_loss_sce
from train.finetune import finetune
from train.joint import joint

from evaluate.softmax import accuracy

from .data import SplitMNIST
from .models import make_state, sr


splitmnist_train = SplitMNIST()
splitmnist_test = SplitMNIST(train=False)
state_main_init, state_consolidator_init = make_state(sr.Main(), sr.Consolidator())
state_main = state_main_init
hyperparams = {'precision': 0.1}
for i, dataset in enumerate(tqdm(splitmnist_train)):
    state_main, hyperparams, loss = joint(
        make_loss_sce, 1, 64, state_main, hyperparams, dataset
    )
    print([
        accuracy(None, state_main, dataset)
        for dataset in list(splitmnist_test)[: i + 1]
    ])
