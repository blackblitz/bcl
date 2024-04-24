"""Neural Consolidation."""

from jax import flatten_util, jit, random, vmap
import optax

from . import make_step

def make_loss(state, hyperparams, loss_basic):
    return jit(
        lambda params, x, y:
        hyperparams['state_consolidator'].apply_fn(
            {'params': hyperparams['state_consolidator'].params},
            flatten_util.ravel_pytree(params)[0]
        )[0] + loss_basic(params, x, y)
    )


def make_loss_consolidator_(hyperparams, loss_target):
    pflat, punflatten = flatten_util.ravel_pytree(hyperparams['minimum'])
    ball = random.ball(
        random.PRNGKey(1337), len(pflat), shape=(hyperparams['size'],)
    )
    pflats = pflat + hyperparams['radius'] * ball
    return jit(
        lambda params, x, y:
        optax.l2_loss(
            hyperparams['state_consolidator'].apply_fn(
                {'params': params}, pflats
            )[:, 0],
            vmap(
                loss_target, in_axes=(0, None, None)
            )(vmap(punflatten)(pflats), x, y)
        ).sum()
    )


def make_loss_consolidator(hyperparams, loss_target, batches):
    pflat, punflatten = flatten_util.ravel_pytree(hyperparams['minimum'])
    ball = random.ball(
        random.PRNGKey(1337), len(pflat), shape=(hyperparams['size'],)
    )
    pflats = pflat + hyperparams['radius'] * ball
    loss_true = sum(
        vmap(
            loss_target, in_axes=(0, None, None)
        )(vmap(punflatten)(pflats), x, y)
        for x, y in batches
    )
    return jit(
        lambda params, x, y:
        optax.huber_loss(
            hyperparams['state_consolidator'].apply_fn(
                {'params': params}, pflats
            )[:, 0],
            loss_true
        ).sum()
    )


def update_hyperparams(state, hyperparams, loss_basic, batches):
    hyperparams['minimum'] = state.params
    state_consolidator = hyperparams['state_consolidator']
    loss = make_loss(state, hyperparams, loss_basic)
    loss_consolidator = make_loss_consolidator(hyperparams, loss, batches)
    step_consolidator = make_step(loss_consolidator)
    for _ in range(1000):
        state_consolidator = step_consolidator(state_consolidator, None, None)
    hyperparams['state_consolidator'] = state_consolidator
