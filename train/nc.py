"""Neural Consolidation."""

from flax import linen as nn
from jax import flatten_util, grad, jit, random, tree_util, vmap
import optax
from torch.utils.data import DataLoader

from . import make_loss_reg, make_step
from dataseqs import numpy_collate
from models import TrainState


def make_loss_nc(state, loss):
    return jit(
        lambda params, x, y:
        state.hyperparams['state_consolidator'].apply_fn(
            {'params': state.hyperparams['state_consolidator'].params},
            flatten_util.ravel_pytree(params)[0]
        )[0] + loss(params, x, y)
    )


def make_loss_consolidator(state, loss):
    pflat, punflatten = flatten_util.ravel_pytree(state.hyperparams['minimum'])
    pflats = pflat + state.hyperparams['radius'] * state.hyperparams['ball']
    return jit(
        lambda params, x, y:
        optax.l2_loss(
            state.apply_fn({'params': params}, pflats)[:, 0],
            vmap(
                loss, in_axes=(0, None, None)
            )(vmap(punflatten)(pflats), x, y)
        ).sum()
    )


class Consolidator(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(20)(x)
        x = nn.swish(x)
        x = nn.Dense(1)(x)
        return x


def default_state_consolidator(state):
    consolidator = Consolidator()
    key1, key2 = random.split(random.PRNGKey(1337))
    pflat = flatten_util.ravel_pytree(state.params)[0]
    return TrainState.create(
        apply_fn=consolidator.apply,
        params=consolidator.init(
            key1, jnp.expand_dims(pflat, 0)
        )['params'],
        tx=optax.adam(0.1),
        hyperparams={
            'minimum': tree_util.tree_map(jnp.zeros_like, state.params),
            'radius': 20.0,
            'ball': random.ball(key2, len(pflat), shape=(10000,))
        }
    )


def nc(
    nepochs, state, loss_basic, dataseq, precision=0.1,
    state_consolidator=None, dataloader_kwargs=None
):
    if state_consolidator is None:
        state_consolidator = default_state_consolidator(state)
    state = state.replace(hyperparams={'precision': precision})
    loss = make_loss_reg(state, loss_basic)
    step = make_step(loss)
    for i, dataset in enumerate(dataseq.train()):
        if dataloader_kwargs is None:
            x, y = next(iter(
                DataLoader(
                    dataset,
                    batch_size=len(dataset),
                    collate_fn=numpy_collate
                )
            ))
            for _ in range(nepochs):
                state = step(state, x, y)
        else:
            for _ in range(nepochs):
                for x, y in DataLoader(dataset, **dataloader_kwargs):
                    state = step(state, x, y)
        yield state, loss
        state_consolidator = state_consolidator.replace(
            hyperparams=state_consolidator.hyperparams | {'minimum': state.params}
        )
        loss_consolidator = make_loss_consolidator(state_consolidator, loss)
        step_consolidator = make_step(loss_consolidator)
        if dataloader_kwargs is None:
            x, y = next(iter(
                DataLoader(
                    dataset,
                    batch_size=len(dataset),
                    collate_fn=numpy_collate
                )
            ))
            for _ in range(nepochs):
                state_consolidator = step_consolidator(state_consolidator, x, y)
        else:
            for _ in range(nepochs):
                for x, y in DataLoader(dataset, **dataloader_kwargs):
                    state_consolidator = step_consolidator(state_consolidator, x, y)
        state = state.replace(hyperparams={'state_consolidator': state_consolidator})
        loss = make_loss_nc(state, loss_basic)
        step = make_step(loss)