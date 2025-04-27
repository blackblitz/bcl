# Bayesian Continual Learning

This repository contains implementations of algorithms for Bayesian continual learning, including the official implementation of

1. [On Sequential Maximum a Posteriori Inference for Continual Learning](https://arxiv.org/abs/2405.16498)
2. [Sequential Function-Space Variational Inference via Gaussian Mixture Approximation](https://arxiv.org/abs/2503.07114)

## Setting up

Make sure that python is installed (tested with python 3.11) and use pip to install the requirements:

```
pip install -r requirements.txt
```

Set deterministic GPU operations for reproducibility:

```
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
```

## Running the Experiments

### Preparing the Data

`HAM10000` and `BCN20000` must be manually downloaded and stored under `data/HAM10000` and `data/BCN20000`, respectively, which can be done by using the CLI tool [`isic`](https://github.com/ImageMarkup/isic-cli).

The `bcl.dataprep` script stores the data as `npy` files under `data`:

```
python -m bcl.dataprep [task_sequence]
```

Here is a list of available task sequences: `cisplit2diris`, `disinusoid`, `cisplitiris`, `cisplitwine`, `cisplitmnist`, `cisplitcifar10`, `cisplitham8`, `displitmnist`, `displitcifar8`, `displitham6`, `emnistletters`, `cifar100` and `bcn12`.

### Training

Experiments are specified in toml files under `experiments` with name `[project]_[dataset_sequence]_[variant]`. Experiments in [1] are under project `smi`:

- `smi_cisplit2diris_sr`
- `smi_cisplit2diris_fcnn`
- `smi_cisplitiris_sr`
- `smi_cisplitiris_fcnn`
- `smi_cisplitwine_sr`
- `smi_cisplitwine_fcnn`
- `smi_cisplitmnist_fe`
- `smi_cisplitcifar10_fe`
- `smi_cisplitham8_fe`
- `smi_displitmnist_fe`
- `smi_displitcifar8_fe`
- `smi_displitham6_fe`

Experiments in [2] are under project `svi`:

- `svi_cisplit2diris_fcnn`
- `svi_disinusoid_fcnn`
- `svi_cisplitmnist_fe`
- `svi_cisplitcifar10_fe`
- `svi_cisplitham8_fe`
- `svi_displitmnist_fe`
- `svi_displitcifar8_fe`
- `svi_displitham6_fe`
- `svi_cisplitmnist_fcnn`
- `svi_cisplitmnist_cnn`
- `svi_cisplitcifar10_cnn`
- `svi_cisplitham8_cnn`
- `svi_displitmnist_fcnn`
- `svi_displitmnist_cnn`
- `svi_displitcifar8_cnn`
- `svi_displitham6_cnn`

The `bcl.train` script trains the model using the specified methods and save the model parameters under `results/ckpt`:

```
python -m bcl.train [experiment_name]
```

Joint HMC-NUTS can be run in parallel on multiple CPU cores:

```
JAX_PLATFORM_NAME=cpu XLA_FLAGS="--xla_force_host_platform_device_count=[n_cpu_cores]" python -m bcl.train.hmcnuts [experiment_id] [n_cpu_cores]
```

### Plotting

For visualization experiments, the prediction plots can be produced by using the `bcl.plot` script:

```
python -m bcl.plot [experiment_name]
```

 The `bcl.plot` script saves the plots under `results/plots`. Since joint HMC-NUTS is run separately, its predictions are plotted if its saved model parameters are found.

### Evaluation

The `bcl.evaluate` script saves the evaluation scores in `results/evaluation.jsonl`. Hyperparameter tuning is done by specifying multiple trainers of the same type with different hyperparameters and the script chooses the one with the best validation final average accuracy.

### Reporting

Reports are specified in toml files under `reports` with name `[project]_[group]`. Reports can be generated after evaluation by running the `bcl.report` script:

```
python -m report [report_name]
```

The following reports are available:

- `smi_classical` (`smi` results on classical task sequences)
- `smi_image_fe` (`smi` results on image task sequences with a feature extractor)
- `svi_image_fe` (`svi` results on image task sequences with a feature extractor)
- `svi_image` (`svi` results on image task sequences without a feature extractor)
