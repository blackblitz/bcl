"""Module for computing metametrics."""

import re

from jax import tree_util
import msgpack
import numpy as np


def final_average(metric_vals):
    """Compute the final average metametric."""
    return tree_util.tree_map(lambda x: x[-1].mean().item(), metric_vals)


def madsut1(metric_vals):
    """
    Compute the mean absolute difference with strictly upper triangular ones.

    Intended for use with mean normalized entropy for computing continual
    uncertainty.
    """
    return tree_util.tree_map(
        lambda x: np.abs(x - np.triu(np.ones_like(x), k=1)).mean().item(),
        metric_vals
    )


def read_metric(exp_id, metric_name):
    """Read the metric values from a msgpack file."""
    with open(
        f'results/{exp_id}/{metric_name}.dat',
        mode='rb'
    ) as file:
        metric_vals = msgpack.unpack(file)
    return {re.sub(r'\d', '', k): np.array(v) for k, v in metric_vals.items()}
