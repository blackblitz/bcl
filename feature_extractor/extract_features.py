"""Script that extracts features."""

import argparse
from importlib import import_module
from pathlib import Path

from jax import random
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm

from dataops.array import batch, get_pass_size
from dataops.io import get_filenames, read_task, read_toml, write_toml
from models import ModelSpec, module_map, NLL


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_id', help='experiment ID')
    args = parser.parse_args()

    # read experiment specifications
    exp_path = Path('experiments').resolve()
    exp = read_toml(exp_path / f'{args.experiment_id}.toml')

    # read metadata
    ts_path = Path('data').resolve() / exp['task_sequence']['name']
    metadata = read_toml(ts_path / 'metadata.toml')

    # prepare directory for checkpoints
    result_path = Path('results').resolve() / args.experiment_id
    ckpt_path = result_path / 'ckpt'
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # load checkpoint
    model = getattr(
        import_module(module_map[exp['feature_extractor']['name']]),
        exp['feature_extractor']['name']
    )(**exp['feature_extractor']['args'])
    model_spec = ModelSpec(
        nll=NLL[exp['feature_extractor']['spec']['nll']],
        in_shape=exp['feature_extractor']['spec']['in_shape'],
        out_shape=exp['feature_extractor']['spec']['out_shape']
    )
    path = ckpt_path / 'feature_extractor'
    with ocp.StandardCheckpointer() as ckpter:
        var = ckpter.restore(
            path,
            target=model.init(
                random.key(1337),
                jnp.zeros((1, *model_spec.in_shape)),
                train=False
            )
        )

    # extract features
    path = Path('features').resolve()
    path.mkdir(parents=True, exist_ok=True)
    ocp.test_utils.erase_and_create_empty(path)
    for split in tqdm(['training', 'validation', 'testing'], unit='split'):
        task_ids = tqdm(
            range(1, metadata['length'] + 1), leave=False, unit='task'
        )
        for task_id in task_ids:
            xs, ys = read_task(ts_path, split, task_id)
            xs_filename, ys_filename = get_filenames(split, task_id)
            fe_xs = np.lib.format.open_memmap(
                path / xs_filename, mode='w+',
                dtype=np.float32,
                shape=(len(ys), *exp['model']['spec']['in_shape'])
            )
            fe_ys = np.lib.format.open_memmap(
                path / ys_filename, mode='w+',
                dtype=np.uint8,
                shape=(len(ys),)
            )
            pass_size = get_pass_size(
                exp['feature_extractor']['spec']['in_shape']
            )
            for indices in batch(pass_size, np.arange(len(ys))):
                fe_xs[indices] = np.asarray(model.apply(
                    var, xs[indices],
                    method=lambda module, xs: module.tail(xs, train=False)
                ))
                fe_ys[indices] = ys[indices]
    write_toml(
        metadata | {'input_shape': xs.shape[1:]},
        path / 'metadata.toml'
    )


if __name__ == '__main__':
    main()