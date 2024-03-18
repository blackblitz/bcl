"""Models package."""

from flax.training import train_state


class TrainState(train_state.TrainState):
    hyperparams: dict