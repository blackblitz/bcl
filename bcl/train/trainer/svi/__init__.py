"""Sequential variational inference."""

from jax import random

from .. import OptimizingTrainer
from ...predictor import GaussPredictor, GaussmixPredictor, TPredictor
from ...training import gauss_init, gaussmix_init, t_init
from ...training.vi import gauss, gaussmix, t


class SVI(OptimizingTrainer):
    """Sequential variational inference."""

    def __init__(self, model, mspec, hparams):
        """Initialize self."""
        super().__init__(model, mspec, hparams)
        self.hparams['prior'] = self._init_prior()

    def update_hparams(self, xs, ys):
        """Update the hyperparameters."""
        self.hparams['prior'] = self.state.params


class GaussMixin:
    """Gaussian variational inference mixin."""

    predictor_class = GaussPredictor

    def _init_params(self, key):
        """Initialize the parameters."""
        return gauss_init(key, self.model, self.mspec.in_shape)

    def _init_prior(self):
        """Initialize the prior."""
        return gauss.get_prior(
            self.hparams['precision'], self._init_params(random.key(1337))
        )

    def _sample(self, key, xs=None):
        """Draw a standard sample for the parameter."""
        return gauss.sample(
            key, self.hparams['sample_size'],
            self.hparams['param_example'] if xs is None
            else self.model.apply(
                {'params': self.hparams['param_example']}, xs
            )
        )


class GaussmixMixin:
    """Gaussian mixture variation inference mixin."""

    predictor_class = GaussmixPredictor

    def _init_params(self, key):
        """Initialize the parameters."""
        return gaussmix_init(
            key, self.model,
            self.hparams['n_comp'], self.mspec.in_shape
        )

    def _init_prior(self):
        """Initialize the prior."""
        return gaussmix.get_prior(
            self.hparams['precision'], self._init_params(random.key(1337))
        )

    def _sample(self, key, xs=None):
        """Draw a standard sample for the parameter."""
        return gaussmix.sample(
            key, self.hparams['sample_size'],
            self.hparams['param_example'] if xs is None
            else self.model.apply(
                {'params': self.hparams['param_example']}, xs
            ),
            self.hparams['n_comp']
        )


class TMixin:
    """t variation inference mixin."""

    predictor_class = TPredictor

    def _init_params(self, key):
        """Initialize the state."""
        return t_init(key, self.model, self.mspec.in_shape)

    def _init_prior(self):
        """Initialize the prior."""
        return t.get_prior(
            self.hparams['invscale'], self._init_params(random.key(1337))
        )

    def _sample(self, key, xs=None):
        """Draw a standard sample for the parameter."""
        return t.sample(
            key, self.hparams['sample_size'],
            self.hparams['param_example'] if xs is None
            else self.model.apply(
                {'params': self.hparams['param_example']}, xs
            ),
            self.hparams['df']
        )
