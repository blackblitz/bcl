"""Elastic Weight Consolidation."""

from operator import add

import jax.numpy as jnp
from jax import grad, jit, tree_util, vmap
from jax.lax import scan

from dataio import iter_batches

from .loss import make_loss_reg

from . import Trainer


class ElasticWeightConsolidation(Trainer):
    def update_loss(self):
        if hasattr(self, 'loss'):
            self.loss = jit(
                lambda params, x, y:
                0.5 * self.hyperparams['lambda'] * tree_util.tree_reduce(
                    add,
                    tree_util.tree_map(
                        lambda hess, params, p_min: (
                            hess * (params - p_min) ** 2
                        ).sum(),
                        self.hyperparams['fisher'],
                        params,
                        self.hyperparams['minimum']
                    )
                ) + self.loss_basic(params, x, y)
            )
        else:
            self.loss = make_loss_reg(
                self.state, self.hyperparams['precision'], self.loss_basic
            )


    def _make_fisher(self, x, y, batch_size=1024):
        total = tree_util.tree_map(jnp.zeros_like, self.state.params)
        for x_batch, y_batch in iter_batches(1, batch_size, x, y):
            total = tree_util.tree_map(
                add, total,
                tree_util.tree_map(
                    lambda x: (x ** 2).sum(axis=0),
                    vmap(grad(self.loss_basic), in_axes=(None, 0, 0))(
                        self.state.params,
                        jnp.expand_dims(x_batch, 1),
                        jnp.expand_dims(y_batch, 1)
                    )
                )
            )
        return tree_util.tree_map(
            add,
            self.hyperparams.get(
                'fisher',
                tree_util.tree_map(
                    lambda x: jnp.full_like(x, self.hyperparams['precision']),
                    self.state.params
                )
            ),
            tree_util.tree_map(lambda x: x / len(y), total)
        )


    def update_hyperparams(self, batch_size, x, y):
        self.hyperparams['minimum'] = self.state.params
        self.hyperparams['fisher'] = self._make_fisher(
            x, y, batch_size=batch_size
        )
