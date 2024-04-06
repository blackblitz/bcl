"""Script for Split Iris."""

from copy import deepcopy
from itertools import islice

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from train import make_loss_sce
from train.ah import ah
from train.ewc import ewc
from train.finetune import finetune
from train.joint import joint
from train.nc import nc

from evaluate.softmax import accuracy

from .data import SplitIris
from .models import make_state, nnet, sreg

plt.style.use('bmh')

splitiris = SplitIris()
labels = [
    'Joint training',
    'Fine-tuning',
    'Elastic Weight Consolidation',
    'Autodiff Hessian',
    'Neural Consolidation'
]
algos = [joint, finetune, ewc, ah, nc]
hyperparams_inits = [
    {'precision': 0.1},
    {'precision': 0.1},
    {'init': True, 'precision': 0.1, 'lambda': 1.0},
    {'init': True, 'precision': 0.1},
    {'init': True, 'precision': 0.1, 'radius': 20.0, 'size': 10000}
]
for name, model in zip(['sreg', 'nnet'], [sreg, nnet]):
    state_main_init, state_consolidator_init = make_state(model.Main(), model.Consolidator())
    hyperparams_inits[4]['state_consolidator'] = state_consolidator_init
    for label, algo, hyperparams_init in zip(labels, algos, hyperparams_inits):
        print(label)
        hyperparams = deepcopy(hyperparams_init)
        state_main = state_main_init
        for i, dataset in enumerate(splitiris.train()):
            state_main, hyperparams, loss = algo(
                make_loss_sce, 1000, None, state_main, hyperparams, dataset
            )
            print(np.mean([accuracy(None, state_main, d) for d in islice(splitiris.test(), 0, i + 1)]))
