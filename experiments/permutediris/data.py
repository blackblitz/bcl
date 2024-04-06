"""Permuted Iris."""

import numpy as np
from jax import random

from datasets.iris import Iris


class PermutedIris:
    def train(self):
        key = random.PRNGKey(1337)
        for i in range(3):
            if i == 0:
                yield Iris()
            else:
                key, key1 = random.split(key)
                
                def transform(x):
                    return np.asarray(random.permutation(key1, x))
                    
                yield Iris(transform=transform)

    def test(self):
        key = random.PRNGKey(1337)
        for i in range(3):
            if i == 0:
                yield Iris(train=False)
            else:
                key, key1 = random.split(key)
                
                def transform(x):
                    return np.asarray(random.permutation(key1, x))
                    
                yield Iris(train=False, transform=transform)


