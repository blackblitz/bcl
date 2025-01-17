# Bayesian Continual Learning

This repository contains implementations of algorithms for Bayesian continual learning, including the official implementation of "On Sequential Maximum a Posteriori Inference for Continual Learning" [1].

## Setting up

Make sure that python is installed (tested with python 3.10) and use pip to install the requirements:

```
pip install -r requirements.txt
```

Set deterministic GPU operations for reproducibility:

```
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
```

## Running experiments

Before running experiments, data must be prepared by running the `dataprep` script:

```
python -m dataprep [task_sequence]
```

For the experiments in [1], the following tasks or task sequences are required: `cisplitiris2`, `cisplitiris`, `cisplitwine`, `cisplitmnist`, `cisplitcifar10`, `cisplitham8`, `displitmnist`, `displitcifar8`, `displitham6`, `emnistletters`, `cifar100` and `bcn12`. The `dataprep` prepares the data as `npy` files and store them under `data`. `HAM10000` images and metadata must be manually downloaded before preparing `cisplitham8` and `displitham6`. `BCN20000` images and metadata must be manually downloaded and stored under `data/BCN20000` before preparing `bcn12`. Both of these data collections can be downloaded via the CLI tool [`isic`](https://github.com/ImageMarkup/isic-cli).

Experiments are specified in toml files under `experiments` with name `[project]_[dataset_sequence]_[variant]`. Use the `train` script to train with all the specified methods and use the `evaluate` script to evaluate them:

```
python -m train [experiment_name]
python -m evaluate [experiment_name]
```

For visualization experiments, the prediction plots can be produced by using the `plot` script:

```
python -m plot [experiment_name]
```

The `train` script saves the model parameters under `results/ckpt`. The `evaluate` script saves the evaluation scores in `results/evaluation.jsonl`. The `plot` script saves the plots under `results/plots`.

Hyperparameter tuning is done by specifying multiple trainers of the same type with different hyperparameters and the `evaluate` script chooses the one with the best validation final average accuracy.
