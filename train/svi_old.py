# """Sequential variational inference."""
# 
# from abc import abstractmethod
# from operator import add, sub, truediv
# from pathlib import Path
# 
# from flax.training.train_state import TrainState
# import jax.numpy as jnp
# from jax import jacrev, jit, random, tree_util, vmap
# from jax.nn import softmax, softplus
# from jax.scipy.special import rel_entr
# from optax import adam
# 
# from dataio.dataset_sequences.datasets import dataset_to_arrays
# 
# from .base import ContinualTrainer, UpdateStateMixin
# from .replay import (
#     InitCoresetMixin, RandomCoresetMixin, BalancedRandomCoresetMixin
# )
# from .tree import tree_dot, tree_gauss, tree_size, tree_sum
# 
# 
# def isoftplus(x):
#     """Apply the inverse of the softplus function."""
#     return jnp.where(x > 0., x + jnp.log(-jnp.expm1(-x)), jnp.nan)
# 
# 
# def msd2var(msd):
#     """Convert modified standard deviation to variance."""
#     return tree_util.tree_map(lambda x: softplus(x) ** 2, msd)
# 
# 
# class GaussianSVIMixin:
#     """Mixin for initializing the state for Gaussian SVI."""
# 
#     def _init_state(self):
#         """Initialize the state."""
#         key1, key2 = random.split(
#             random.PRNGKey(self.hyperparams['init_state_seed'])
#         )
#         mean = self.model.init(
#             key1, jnp.zeros(self.hyperparams['input_shape'])
#         )['params']
#         msd = self.model.init(
#             key2, jnp.zeros(self.hyperparams['input_shape'])
#         )['params']
#         return TrainState.create(
#             apply_fn=self.model.apply,
#             params={'mean': mean, 'msd': msd},
#             tx=adam(self.hyperparams['lr'])
#         )
# 
#     def _init_hyperparams(self):
#         """Initialize the hyperparameters."""
#         return {
#             'mean': tree_util.tree_map(
#                 jnp.zeros_like, self.state.params['mean']
#             ),
#             'msd': tree_util.tree_map(
#                 lambda x: jnp.full_like(
#                     x, isoftplus(jnp.sqrt(1 / self.hyperparams['precision']))
#                 ), self.state.params['msd']
#             )
#         }
# 
#     @classmethod
#     def sample_std(cls, seed, sample_size, var_params):
#         """Generate a standard sample of the parameters."""
#         return tree_gauss(
#             random.PRNGKey(seed),
#             sample_size,
#             var_params['mean']
#         )
# 
#     @classmethod
#     def sample_params(cls, var_params, std_sample):
#         """Generate a variational sample of the parameters."""
#         return vmap(
#             lambda zs: tree_util.tree_map(
#                 lambda m, r, z: m + softplus(r) * z,
#                 var_params['mean'], var_params['msd'], zs
#             )
#         )(std_sample)
# 
# 
# class GaussianMixtureSVIMixin:
#     """Mixin for initializing the state for Gaussian-mixture SVI."""
# 
#     def _init_state(self):
#         """Initialize the state."""
#         key1, key2 = random.split(
#             random.PRNGKey(self.hyperparams['init_state_seed'])
#         )
#         mean = vmap(
#             lambda key: self.model.init(
#                 key, jnp.zeros(self.hyperparams['input_shape'])
#             )['params']
#         )(random.split(key1, self.hyperparams['n_comp']))
#         msd = vmap(
#             lambda key: self.model.init(
#                 key, jnp.zeros(self.hyperparams['input_shape'])
#             )['params']
#         )(random.split(key2, self.hyperparams['n_comp']))
#         return TrainState.create(
#             apply_fn=self.model.apply,
#             params={
#                 'logit': jnp.zeros(self.hyperparams['n_comp']),
#                 'mean': mean, 'msd': msd
#             },
#             tx=adam(self.hyperparams['lr'])
#         )
# 
#     def _init_hyperparams(self):
#         """Initialize the hyperparameters."""
#         return {
#             'logit': jnp.zeros(self.hyperparams['n_comp']),
#             'mean': tree_util.tree_map(
#                 jnp.zeros_like, self.state.params['mean']
#             ),
#             'msd': tree_util.tree_map(
#                 lambda x: jnp.full_like(
#                     x, isoftplus(jnp.sqrt(1 / self.hyperparams['precision']))
#                 ), self.state.params['msd']
#             )
#         }
# 
#     @classmethod
#     def sample_std(cls, seed, sample_size, var_params):
#         """Generate a standard sample of the parameters."""
#         key1, key2 = random.split(random.PRNGKey(seed))
#         return {
#             'gauss': tree_gauss(key1, sample_size, var_params['mean']),
#             'gumbel': random.gumbel(
#                 key2, (
#                     sample_size,
#                     tree_util.tree_leaves(var_params['mean'])[0].shape[0]
#                 )
#             )
#         }
# 
#     @classmethod
#     def sample_params(cls, var_params, std_sample):
#         """Generate a variational sample of the parameters."""
#         prob = softmax(1000.0 * (var_params['logit'] + std_sample['gumbel']))
#         comp = vmap(
#             lambda zs: tree_util.tree_map(
#                 lambda m, r, z: m + softplus(r) * z,
#                 var_params['mean'], var_params['msd'], zs
#             )
#         )(std_sample['gauss'])
#         return vmap(
#             lambda x, y: tree_util.tree_map(
#                 lambda z: jnp.tensordot(z, x, axes=(0, 0)), y
#             )
#         )(prob, comp)
# 
# 
# class SVI(UpdateStateMixin, ContinualTrainer):
#     """Abstract class for sequential variational inference."""
# 
#     def __init__(self, model, make_predictor, hyperparams):
#         """Initialize self."""
#         super().__init__(model, make_predictor, hyperparams)
#         self.state = self._init_state()
#         self.hyperparams |= self._init_hyperparams()
#         self._std_sample = self.sample_std(
#             self.hyperparams['sample_seed'],
#             self.hyperparams['sample_size'],
#             self.state.params
#         )
# 
#     def train(self, dataset):
#         """Train self."""
#         self._update_loss(dataset)
#         path = Path('data.npy')
#         xs, ys = dataset_to_arrays(dataset, path)
#         self._update_state(xs, ys)
#         path.unlink()
#         self.hyperparams |= self.state.params
# 
#     @abstractmethod
#     def _init_state(self):
#         """Initialize the state."""
# 
#     @abstractmethod
#     def _init_hyperparams(self):
#         """Initialize the hyperparams."""
# 
#     @classmethod
#     @abstractmethod
#     def sample_std(cls, sample_seed, sample_size, var_params):
#         """Generate a standard sample of the parameters."""
# 
#     @classmethod
#     @abstractmethod
#     def sample_params(cls, var_params, std_sample):
#         """Generate a variational sample of the parameters."""
# 
#     @abstractmethod
#     def _update_loss(self, dataset):
#         """Update loss function."""
# 
#     def _eloss(self, params, xs, ys):
#         """Compute the expected loss."""
#         return vmap(
#             self.hyperparams['static']['basic_loss_fn'], in_axes=(0, None, None)
#         )(self.sample_params(params, self._std_sample), xs, ys).mean()
# 
# 
# class VCL(SVI):
#     """Variational continual learning."""
# 
#     def _update_loss(self, dataset):
#         """Update loss."""
#         n_batches = -(len(dataset) // -self.hyperparams['draw_batch_size'])
#         self.loss_fn = (
#             lambda params, xs, ys: (
#                 self._eloss(params, xs, ys)
#                 + self.hyperparams['beta'] / n_batches
#                 * self._kldiv(params, self.hyperparams)
#             )
#         )
# 
#     @classmethod
#     @abstractmethod
#     def _kldiv(cls, params, hyperparams):
#         """Compute the KL divergence."""
# 
# 
# class GaussianVCL(GaussianSVIMixin, VCL):
#     """Gaussian variational continual learning."""
# 
#     @classmethod
#     def _kldiv(cls, params, hyperparams):
#         """Compute the KL divergence."""
#         mean_var = params['mean']
#         var_var = msd2var(params['msd'])
#         mean_prior = hyperparams['mean']
#         var_prior = msd2var(hyperparams['msd'])
#         return 0.5 * (
#             tree_sum(tree_util.tree_map(truediv, var_var, var_prior))
#             - tree_size(mean_var)
#             + tree_sum(tree_util.tree_map(
#                 lambda m1, m2, v: (m1 - m2) ** 2 / v,
#                 mean_prior, mean_var, var_prior
#             )) + tree_sum(tree_util.tree_map(jnp.log, var_prior))
#             - tree_sum(tree_util.tree_map(jnp.log, var_var))
#         )
# 
# 
# class GaussianMixtureVCL(GaussianMixtureSVIMixin, VCL):
#     """Gaussian-mixture variational continual learning."""
# 
#     @classmethod
#     def _kldiv(cls, params, hyperparams):
#         """Compute the KL divergence."""
#         p_var = softmax(params['logit'])
#         p_prior = softmax(hyperparams['logit'])
#         d_cat = rel_entr(p_var, p_prior).sum()
#         d_gauss = vmap(GaussianVCL._kldiv)(
#             {'mean': params['mean'], 'msd': params['msd']},
#             {'mean': hyperparams['mean'], 'msd': hyperparams['msd']},
#         )
#         return d_cat + p_var @ d_gauss
# 
# 
# class SFSVI(SVI, InitCoresetMixin):
#     """Abstract class for sequential function-space variational inference."""
# 
#     def __init__(self, model, make_predictor, hyperparams):
#         """Initialize self."""
#         super().__init__(model, make_predictor, hyperparams)
#         self.coreset = (
#             self._init_coreset() if self.hyperparams['init_coreset'] else []
#         )
# 
#     def train(self, dataset):
#         """Train self."""
#         super().train(dataset)
#         self._update_coreset(dataset)
# 
#     def _update_loss(self, dataset):
#         """Update loss."""
#         path = Path('core.npy')
#         core, _ = dataset_to_arrays(self.coreset, path)
#         path.unlink()
#         n_batches = -(len(dataset) // -self.hyperparams['draw_batch_size'])
#         self.loss_fn = jit(
#             lambda params, xs, ys: (
#                 self._eloss(params, xs, ys)
#                 + self.hyperparams['beta'] / n_batches * self._kldiv(
#                     params, self.hyperparams, self.state.apply_fn, core
#                 )
#             )
#         )
# 
#     @classmethod
#     @abstractmethod
#     def _kldiv(cls, params, hyperparams, state, core):
#         """Compute the KL divergence."""
# 
# 
# class GaussianSFSVIMixin:
#     """Gaussian sequential function-space variational inference."""
# 
#     @classmethod
#     def _fvar(cls, jac, var):
#         """Compute the function-space variance."""
#         return vmap(vmap(
#             lambda var, jac: tree_util.tree_reduce(
#                 add,
#                 tree_util.tree_map(
#                     lambda v, j: (v * j ** 2).sum(), var, jac
#                 )
#             ),
#             in_axes=(None, 0)
#         ), in_axes=(None, 0))(var, jac)
# 
#     @classmethod
#     def _kldiv(cls, params, hyperparams, apply, core):
#         """Compute KL divergence."""
#         mean_var = params['mean']
#         var_var = msd2var(params['msd'])
#         mean_prior = hyperparams['mean']
#         var_prior = msd2var(hyperparams['msd'])
#         jac_at = jacrev(lambda params: apply({'params': params}, core))
#         fmean_var = apply({'params': mean_var}, core)
#         fmean_prior = apply({'params': mean_prior}, core)
#         fvar_var = cls._fvar(jac_at(mean_var), var_var)
#         fvar_prior = cls._fvar(jac_at(mean_prior), var_prior)
#         fvar_ratio = fvar_var / fvar_prior
#         return 0.5 * (
#             (fmean_var - fmean_prior) ** 2 / fvar_prior - 1
#             - jnp.log(fvar_ratio) + fvar_ratio
#         ).mean(axis=0).sum()
# 
# 
# class GaussianMixtureSFSVIMixin:
#     """Gaussian-mixture sequential function-space variational inference."""
# 
#     @classmethod
#     def _fmean(cls, apply, jac, center, mean, core):
#         """Compute the function-space mean."""
#         return vmap(
#             lambda m:
#             apply({'params': center}, core)
#             + vmap(tree_dot, in_axes=(1, None))(
#                 jac, tree_util.tree_map(sub, m, center)
#             )
#         )(mean)
# 
#     @classmethod
#     def _fvar(cls, jac, var):
#         """Compute the function-space variance."""
#         return vmap(GaussianSFSVIMixin._fvar, in_axes=(None, 0))(jac, var)
# 
#     @classmethod
#     def _kldiv(cls, params, hyperparams, apply, core):
#         """Compute KL divergence."""
#         mean_var = params['mean']
#         center_var = tree_util.tree_map(lambda x: x.mean(axis=0), mean_var)
#         var_var = msd2var(params['msd'])
#         mean_prior = hyperparams['mean']
#         center_prior = tree_util.tree_map(
#             lambda x: x.mean(axis=0), mean_prior
#         )
#         var_prior = msd2var(hyperparams['msd'])
#         jac_at = jacrev(lambda params: apply({'params': params}, core))
#         jac_var = jac_at(center_var)
#         jac_prior = jac_at(center_prior)
#         fmean_var = cls._fmean(apply, jac_var, center_var, mean_var, core)
#         fmean_prior = cls._fmean(
#             apply, jac_prior, center_prior, mean_prior, core
#         )
#         fvar_var = cls._fvar(jac_var, var_var)
#         fvar_prior = cls._fvar(jac_prior,  var_prior)
#         fvar_ratio = fvar_var / fvar_prior
#         p_var = softmax(params['logit'])
#         p_prior = softmax(hyperparams['logit'])
#         d_cat = rel_entr(p_var, p_prior).sum()
#         d_gauss = 0.5 * (
#             (fmean_var - fmean_prior) ** 2 / fvar_prior - 1
#             - jnp.log(fvar_ratio) + fvar_ratio
#         )
#         return (
#             d_cat + jnp.tensordot(p_var, d_gauss, axes=(0, 0))
#         ).mean(axis=0).sum()
# 
# 
# class RandomCoresetGaussianSFSVI(
#     GaussianSFSVIMixin, RandomCoresetMixin, GaussianSVIMixin, SFSVI
# ):
#     """Gaussian S-FSVI with random coreset."""
# 
# 
# class BalancedRandomCoresetGaussianSFSVI(
#     GaussianSFSVIMixin, BalancedRandomCoresetMixin, GaussianSVIMixin, SFSVI
# ):
#     """Gaussian S-FSVI with balanced random coreset."""
# 
# 
# class BalancedRandomCoresetGaussianMixtureSFSVI(
#     GaussianMixtureSFSVIMixin, BalancedRandomCoresetMixin,
#     GaussianMixtureSVIMixin, SFSVI
# ):
#     """Gaussian-mixture S-FSVI with balanced random coreset."""