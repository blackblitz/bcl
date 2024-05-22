# Bayesian Continual Learning

This repository contains the official implementation of "On Sequential Loss Approximation for Continual Learning".

## Setting up

Make sure that python is installed (tested with python 3.10) and use pip to install the requirements:

```
pip install -r requirements.txt
```

Then, create directories `plots` and `results`:

```
mkdir plots results
```

Finally, set deterministic GPU operations for reproducibility:

```
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
```

## Running experiments

Split Iris 1 and Split Iris 2 produce visualizations, which are stored as `png` images under `plots`.

```
python -m experiments.splitiris1
python -m experiments.splitiris2
```

Split Iris, Split Wine, Split MNIST with pre-training and Split CIFAR-10 with pre-training produce results, which are stored as `msgpack` data files under `results`:

```
python -m experiments.splitiris
python -m experiments.splitwine
python -m experiments.pretrained_splitmnist.pretrain
python -m experiments.pretrained_splitmnist
python -m experiments.pretrained_splitcifar10.pretrain
python -m experiments.pretrained_splitcifar10
```

The results can be viewed by running `view.py`:

```
python view.py -a results/splitiris.dat
python view.py -a results/splitwine.dat
python view.py -a results/pretrained_splitmnist.dat
python view.py -a results/pretrained_splitcifar10.dat
```

`npy` files are produced during the experiments. They are used only for memory-mapped reading and can be deleted after the experiments are complete.

All experiments are reproducible if deterministic GPU operations are used (see above). Data shuffling, parameter initialization and sampling in neural consolidation all use `jax.random` with a fixed seed for pseudo-random-number generation.
