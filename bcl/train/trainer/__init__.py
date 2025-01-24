"""Trainer."""

from abc import ABC, abstractmethod
from pathlib import Path

from flax.training.train_state import TrainState
from jax import random
import jax.numpy as jnp
from optax import adam, constant_schedule, cosine_onecycle_schedule

from ..training import init
from ..training.loss import get_nll

from ...dataops.array import get_n_batches, get_pass_size

coreset_memmap_path = Path('coreset.memmap').resolve()
coreset_zarr_path = Path('coreset.zarr').resolve()
sampling = ['HMCNUTS']
smi_simple = [
    'AutodiffQuadraticConsolidation', 'Finetuning',
    'ElasticWeightConsolidation', 'Joint',
    'NeuralConsolidation', 'SynapticIntelligence'
]
smi_replay = ['GDumb', 'TICReplay']
svi_simple = [
    'SimpleGMVCL', 'SimpleGVCL', 'SimpleGMSFSVI',
    'SimpleGSFSVI', 'SimpleTSFSVI', 'SimpleTVCL'
]
svi_replay = [
    'PriorExactGMSFSVI', 'PriorExactGSFSVI', 'PriorExactTSFSVI',
    'LikelihoodExactGMSFSVI', 'LikelihoodExactGSFSVI', 'LikelihoodExactTSFSVI',
    'LikelihoodExactGMVCL', 'LikelihoodExactGVCL', 'LikelihoodExactTVCL'
]

module_map = (
    dict.fromkeys(sampling, 'bcl.train.trainer.sampling')
    | dict.fromkeys(smi_simple, 'bcl.train.trainer.smi.simple')
    | dict.fromkeys(smi_replay, 'bcl.train.trainer.smi.replay')
    | dict.fromkeys(svi_simple, 'bcl.train.trainer.svi.simple')
    | dict.fromkeys(svi_replay, 'bcl.train.trainer.svi.replay')
)


class OptimizingTrainer(ABC):
    """Abstract base class for an optimizing trainer."""

    def __init__(self, model, mspec, hparams):
        """Intialize self."""
        self.model = model
        self.mspec = mspec
        self.hparams = hparams
        self.state = None
        self.loss = None
        key_names = ['update_loss', 'update_state', 'update_hparams']
        self.hparams |= {
            'keys': dict(zip(
                key_names,
                random.split(
                    random.key(self.hparams['seed']),
                    num=len(key_names)
                )
            )),
            'pass_size': get_pass_size(self.mspec.in_shape),
            'param_example': init(
                random.key(1337), self.model, self.mspec.in_shape
            ),
            'nll': get_nll(self.mspec.nll)(
                self.model.apply,
                self.mspec.cscale / jnp.array(self.mspec.cratio)
            )
        }

    def _make_lr_schedule(self, size):
        """Return the learning rate schedule."""
        match self.hparams['lr_schedule']:
            case 'constant':
                return constant_schedule(self.hparams['base_lr'])
            case 'onecycle':
                return cosine_onecycle_schedule(
                    transition_steps=(
                        self.hparams['n_epochs'] * get_n_batches(
                            size, self.hparams['batch_size']
                        )
                    ),
                    peak_value=self.hparams['base_lr']
                )

    def _init_state(self, key, size):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=(
                self._init_params(key)
                if self.state is None
                else self.state.params
            ),
            tx=adam(self._make_lr_schedule(size))
        )

    @abstractmethod
    def update_loss(self, xs, ys):
        """Update the loss function."""

    @abstractmethod
    def update_state(self, xs, ys):
        """Update the training state."""

    @abstractmethod
    def update_hparams(self, xs, ys):
        """Update the hyperparameters."""

    def train(self, xs, ys):
        """Train with a dataset."""
        self.update_loss(xs, ys)
        yield from self.update_state(xs, ys)
        self.update_hparams(xs, ys)


class SamplingTrainer(ABC):
    """Abstract base class for a sampling trainer."""

    def __init__(self, model, mspec, hparams):
        """Intialize self."""
        self.model = model
        self.mspec = mspec
        self.hparams = hparams
        self.sample = None
        key_names = ['update_sample', 'update_hparams']
        self.hparams |= {
            'keys': dict(zip(
                key_names,
                random.split(
                    random.key(self.hparams['seed']),
                    num=len(key_names)
                )
            )),
            'pass_size': get_pass_size(self.mspec.in_shape),
            'param_example': init(
                random.key(1337), self.model, self.mspec.in_shape
            ),
            'nll': get_nll(self.mspec.nll)(
                self.model.apply,
                self.mspec.cscale / jnp.array(self.mspec.cratio)
            )
        }

    @abstractmethod
    def update_sample(self, xs, ys):
        """Update the sample."""

    @abstractmethod
    def update_hparams(self, xs, ys):
        """Update the hyperparameters."""

    def train(self, xs, ys):
        """Train with a dataset."""
        self.update_sample(xs, ys)
        self.update_hparams(xs, ys)
