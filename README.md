# Bayesian Continual Learning

This repository contains implementations of algorithms for Bayesian continual learning, including the official implementation of "On Sequential Maximum a Posteriori Inference for Continual Learning" [1] and "Sequential Function-Space Variational Inference via Gaussian Mixture Approximation" [2].

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

For the experiments in [1], the following tasks or task sequences are required: `cisplit2diris`, `cisplitiris`, `cisplitwine`, `cisplitmnist`, `cisplitcifar10`, `cisplitham8`, `displitmnist`, `displitcifar8`, `displitham6`, `emnistletters`, `cifar100` and `bcn12`. For the experiments in [2], the following tasks or task sequences are required: `cisplit2diris`, `cisplitmnist`, `cisplitcifar10`, `cisplitham8`, `displitmnist`, `displitcifar8`, `displitham6`, `emnistletters`, `cifar100` and `bcn12`.

`dataprep` prepares the data as `npy` files and stores them under `data`. `HAM10000` images and metadata must be manually downloaded before preparing `cisplitham8` and `displitham6`. `BCN20000` images and metadata must be manually downloaded and stored under `data/BCN20000` before preparing `bcn12`. Both of these data collections can be downloaded via the CLI tool [`isic`](https://github.com/ImageMarkup/isic-cli).

Experiments are specified in toml files under `experiments` with name `[project]_[dataset_sequence]_[variant]`. Experiments in [1] are under project `smi`, while those in [2] are under project `svi`. Use the `train` script to train with all the specified methods and use the `evaluate` script to evaluate them:

```
python -m train [experiment_name]
python -m evaluate [experiment_name]
```

Joint HMC-NUTS can be run in parallel on multiple CPU cores:

```
JAX_PLATFORM_NAME=cpu XLA_FLAGS="--xla_force_host_platform_device_count=[n_cpu_cores]" python -m bcl.train.hmcnuts [experiment_id] [n_cpu_cores]
```

For visualization experiments, the prediction plots can be produced by using the `plot` script:

```
python -m plot [experiment_name]
```

The `train` script and the `train.hmcnuts` script save the model parameters under `results/ckpt`. The `evaluate` script saves the evaluation scores in `results/evaluation.jsonl`. The `plot` script saves the plots under `results/plots`. Since joint HMC-NUTS is run separately, its predictions are plotted if its saved model parameters are found.

Hyperparameter tuning is done by specifying multiple trainers of the same type with different hyperparameters and the `evaluate` script chooses the one with the best validation final average accuracy.

Report specifications are specified in toml files under `reports` with name `[project]_[group]`. Reports can be generated by running the `reports` script:

```
python -m report [report_name]
```
