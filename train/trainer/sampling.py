"""Sampling trainer."""

import blackjax
from jax import devices, device_put, jit, lax, pmap, random, tree_util, vmap
import jax.numpy as jnp

from . import SamplingTrainer
from .coreset import JointCoreset
from ..predictor import BMAPredictor
from ..training import init
from ..training.loss import l2_reg


class HMCNUTS(SamplingTrainer):
    """HMC-NUTS sampling trainer."""

    predictor_class = BMAPredictor

    def __init__(self, model, mspec, hparams):
        """Initialize self."""
        super().__init__(model, mspec, hparams)
        self.hparams['coreset'] = JointCoreset(
            'coreset.zarr', 'coreset.memmap', self.mspec
        )


    def update_sample(self, xs, ys):
        """Update the sample."""
        key1, key2, key3 = random.split(
            self.hparams['keys']['update_sample'], num=3
        )
        self.hparams['coreset'].update(xs, ys)
        self.hparams['coreset'].create_memmap()
        xs, ys = (
            jnp.array(self.hparams['coreset'].memmap['xs']),
            jnp.array(self.hparams['coreset'].memmap['ys'])
        )
        self.hparams['coreset'].delete_memmap()

        @jit
        def logpdf(params):
            return -l2_reg(
                1.0, self.hparams['precision'], self.hparams['nll']
            )(params, xs, ys)

        (states, params), _ = pmap(
            blackjax.window_adaptation(blackjax.nuts, logpdf).run
        )(
            random.split(key1, num=self.hparams['n_chains']),
            vmap(init, in_axes=(0, None, None))(
                random.split(key2, num=self.hparams['n_chains']),
                self.model, self.mspec.in_shape
            )
        )

        kernel = blackjax.nuts.build_kernel()

        if self.hparams['final_only']:

            @jit
            def step(key, state, param):
                state, _ = kernel(key, state, logpdf, **param)
                return state

            for key in random.split(
                key3,
                num=self.hparams['n_steps']
            ):
                states = pmap(step)(
                    random.split(key, num=self.hparams['n_chains']),
                    states, params
                )
            self.sample = device_put(states.position, device=devices()[0])

        else:

            @jit
            def step(state_param, key):
                state, param = state_param
                state, _ = kernel(key, state, logpdf, **param)
                return (state, param), state

            self.sample = tree_util.tree_map(
                lambda x: x.reshape((-1, *x.shape[2:])),
                pmap(
                    lambda sp, k: lax.scan(
                        step, sp,
                        xs=random.split(k, num=self.hparams['n_steps'])
                    )[1]
                )(
                    (states, params),
                    random.split(key4, num=self.hparams['n_chains'])
                ).position
            )

    def update_hparams(self, xs, ys):
        """Update the hyperparameters."""
