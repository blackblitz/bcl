"""Neural Consolidation."""

from jax import flatten_util, jit, random, vmap
import optax

from torchds import fetch
from . import make_loss_reg, make_step


def make_loss_nc(state, hyperparams, loss_basic):
    return jit(
        lambda params, x, y:
        hyperparams['state_consolidator'].apply_fn(
            {'params': hyperparams['state_consolidator'].params},
            flatten_util.ravel_pytree(params)[0]
        )[0] + loss_basic(params, x, y)
    )


def make_loss_consolidator_(state, hyperparams, loss_target):
    pflat, punflatten = flatten_util.ravel_pytree(hyperparams['minimum'])
    ball = random.ball(random.PRNGKey(1337), len(pflat), shape=(hyperparams['size'],))
    pflats = pflat + hyperparams['radius'] * ball
    return jit(
        lambda params, x, y:
        optax.l2_loss(
            state.apply_fn({'params': params}, pflats)[:, 0],
            vmap(
                loss_target, in_axes=(0, None, None)
            )(vmap(punflatten)(pflats), x, y)
        ).sum()
    )


def make_loss_consolidator(state, hyperparams, loss_target, dataset, batch_size):
    pflat, punflatten = flatten_util.ravel_pytree(hyperparams['minimum'])
    ball = random.ball(random.PRNGKey(1337), len(pflat), shape=(hyperparams['size'],))
    pflats = pflat + hyperparams['radius'] * ball
    loss_true = sum(
        vmap(
            loss_target, in_axes=(0, None, None)
        )(vmap(punflatten)(pflats), x, y)
        for x, y in fetch(dataset, 1, batch_size)
    )
    return jit(
        lambda params, x, y:
        optax.l2_loss(
            state.apply_fn({'params': params}, pflats)[:, 0],
            loss_true
        ).sum()
    )


def nc(
    make_loss_basic, num_epochs, batch_size,
    state, hyperparams, dataset, apply=lambda x: x
):
    if hyperparams['init']:
        make_loss = make_loss_reg
    else:
        make_loss = make_loss_nc
    loss_basic = make_loss_basic(state)
    loss = make_loss(state, hyperparams, loss_basic)
    step = make_step(loss)
    for x, y in fetch(dataset, num_epochs, batch_size):
        state = step(state, apply(x), y)
    hyperparams['minimum'] = state.params
    state_consolidator = hyperparams['state_consolidator']
    loss_consolidator = make_loss_consolidator(
        hyperparams['state_consolidator'], hyperparams, loss,
        dataset, 1024
    )
    step_consolidator = make_step(loss_consolidator)
    for _ in range(num_epochs):
        state_consolidator = step_consolidator(state_consolidator, None, None)
    hyperparams['state_consolidator'] = state_consolidator
    hyperparams['init'] = False
    return state, hyperparams, loss
