"""Stateless predictor."""

from jax import nn, random, vmap
import jax.numpy as jnp
from jax.scipy.special import entr
import orbax.checkpoint as ocp

from models import NLL

from .training import init, gauss_init, gaussmix_init, t_init
from .training.vi import gauss, gaussmix, t


def bern_entr(p):
    """Calculate the entropy of Bernoulli random variables."""
    return entr(p) + entr(1 - p)


def cat_entr(p):
    """Calculate the entropy of categorical random variables."""
    return entr(p).sum(axis=-1)


class MAPPredictor:
    """Maximum-a-posteriori predictor."""

    def __init__(self, model, mspec, hparams, params):
        """Initialize self."""
        self.model = model
        self.mspec = mspec
        self.hparams = hparams
        self.params = params

    @classmethod
    def from_checkpoint(cls, model, mspec, hparams, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path,
                target=init(
                    random.PRNGKey(1337), model, mspec.in_shape
                )
            )
        return cls(model, mspec, hparams, params)

    def apply(self, xs):
        """Apply on the data."""
        return self.model.apply({'params': self.params}, xs)

    def __call__(self, xs, decide=True):
        """Predict the data."""
        match self.mspec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                proba = nn.sigmoid(self.apply(xs)[:, 0])
                return proba >= 0.5 if decide else proba
            case NLL.SOFTMAX_CROSS_ENTROPY:
                proba = nn.softmax(self.apply(xs))
                return proba.argmax(axis=-1) if decide else proba

    def entropy(self, xs):
        """Calculate the entropy of predictions."""
        match self.mspec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                return bern_entr(self(xs, decide=False))
            case NLL.SOFTMAX_CROSS_ENTROPY:
                return cat_entr(self(xs, decide=False))


class BMAPredictor:
    """Bayesian-model-averaging predictor."""

    def __init__(self, model, mspec, hparams, param_sample):
        """Initialize self."""
        self.model = model
        self.mspec = mspec
        self.hparams = hparams
        self.param_sample = param_sample

    @classmethod
    def from_checkpoint(cls, model, mspec, hparams, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            target = init(
                random.PRNGKey(1337), model, mspec.in_shape
            )
            param_sample = ckpter.restore(
                path,
                target=vmap(
                    lambda x: target
                )(jnp.arange(hparams['sample_size']))
            )
        return cls(model, mspec, hparams, param_sample)

    def apply(self, params, xs):
        """Apply on the data."""
        return self.model.apply({'params': params}, xs)

    def sample(self, xs):
        """Return prediction samples."""
        match self.mspec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                return vmap(
                    lambda params: nn.sigmoid(self.apply(params, xs)[:, 0])
                )(self.param_sample)
            case NLL.SOFTMAX_CROSS_ENTROPY:
                return vmap(
                    lambda params: nn.softmax(self.apply(params, xs))
                )(self.param_sample)

    def __call__(self, xs, decide=True):
        """Predict the data."""
        match self.mspec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                proba = self.sample(xs).mean(axis=0)
                return proba >= 0.5 if decide else proba
            case NLL.SOFTMAX_CROSS_ENTROPY:
                proba = self.sample(xs).mean(axis=0)
                return proba.argmax(axis=-1) if decide else proba

    def entropy(self, xs):
        """Calculate the entropy of predictions."""
        match self.mspec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                return bern_entr(self.sample(xs).mean(axis=0))
            case NLL.SOFTMAX_CROSS_ENTROPY:
                return cat_entr(self.sample(xs).mean(axis=0))

    def mutual_information(self, xs):
        """Calculate the MI between the predictions and the parameters."""
        match self.mspec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                return (
                    bern_entr(self.sample(xs).mean(axis=0))
                    - bern_entr(self.sample(xs)).mean(axis=0)
                )
            case NLL.SOFTMAX_CROSS_ENTROPY:
                return (
                    cat_entr(self.sample(xs).mean(axis=0))
                    - cat_entr(self.sample(xs)).mean(axis=0)
                )


class GaussPredictor(BMAPredictor):
    """Gaussian variational inference predictor."""

    def __init__(self, model, mspec, hparams, params):
        """Initialize self."""
        param_sample = gauss.transform(
            gauss.get_param(params),
            gauss.sample(
                random.PRNGKey(hparams['seed']),
                hparams['sample_size'],
                init(random.PRNGKey(1337), model, mspec.in_shape)
            )
        )
        super().__init__(model, mspec, hparams, param_sample)

    @classmethod
    def from_checkpoint(cls, model, mspec, hparams, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path, target=gauss_init(
                    random.PRNGKey(1337), model, mspec.in_shape
                )
            )
        return cls(model, mspec, hparams, params)


class GaussmixPredictor(BMAPredictor):
    """Gaussian mixture variational inference predictor."""

    def __init__(self, model, mspec, hparams, params):
        """Initialize self."""
        param_sample = gaussmix.transform(
            gaussmix.get_param(params),
            gaussmix.sample(
                random.PRNGKey(hparams['seed']),
                hparams['sample_size'],
                init(random.PRNGKey(1337), model, mspec.in_shape),
                hparams['n_comp']
            )
        )
        super().__init__(model, mspec, hparams, param_sample)

    @classmethod
    def from_checkpoint(cls, model, mspec, hparams, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path, target=gaussmix_init(
                    random.PRNGKey(1337),
                    model,
                    hparams['n_comp'],
                    mspec.in_shape,
                )
            )
        return cls(model, mspec, hparams, params)


class TPredictor(BMAPredictor):
    """t variational inference predictor."""

    def __init__(self, model, mspec, hparams, params):
        """Initialize self."""
        param_sample = t.transform(
            t.get_param(params, hparams['df']),
            t.sample(
                random.PRNGKey(hparams['seed']),
                hparams['sample_size'],
                init(random.PRNGKey(1337), model, mspec.in_shape),
                hparams['df']
            )
        )
        super().__init__(model, mspec, hparams, param_sample)

    @classmethod
    def from_checkpoint(cls, model, mspec, hparams, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path, target=t_init(
                    random.PRNGKey(1337), model, mspec.in_shape
                )
            )
        return cls(model, mspec, hparams, params)
