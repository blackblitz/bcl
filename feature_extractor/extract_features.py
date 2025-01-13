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
from train.training import init


def extract_features(exp_id, exp_spec):
    """Extract features."""
    # read metadata
    ts_path = Path('data/prepped').resolve() / exp_spec['task_sequence']['name']
    metadata = read_toml(ts_path / 'metadata.toml')

    # prepare directory for checkpoints
    result_path = Path('results').resolve() / exp_id
    ckpt_path = result_path / 'ckpt'
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # load checkpoint
    model = getattr(
        import_module(module_map[exp_spec['feature_extractor']['name']]),
        exp_spec['feature_extractor']['name']
    )(**exp_spec['feature_extractor']['args'])
    mspec = ModelSpec(
        nll=NLL[exp_spec['feature_extractor']['spec']['nll']],
        in_shape=exp_spec['feature_extractor']['spec']['in_shape'],
        out_shape=exp_spec['feature_extractor']['spec']['out_shape'],
        cweight=exp_spec['feature_extractor']['spec']['cweight']
    )
    path = ckpt_path / 'feature_extractor'
    with ocp.StandardCheckpointer() as ckpter:
        params = ckpter.restore(
            path,
            target=init(random.key(1337), model, mspec.in_shape)
        )

    # extract features
    path = Path('data/features').resolve()
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
                shape=(len(ys), *exp_spec['model']['spec']['in_shape'])
            )
            fe_ys = np.lib.format.open_memmap(
                path / ys_filename, mode='w+',
                dtype=np.uint8,
                shape=(len(ys),)
            )
            pass_size = get_pass_size(
                exp_spec['feature_extractor']['spec']['in_shape']
            )
            for indices in batch(pass_size, np.arange(len(ys))):
                fe_xs[indices] = np.asarray(model.apply(
                    {'params': params}, xs[indices],
                    method=lambda module, xs: module.tail(xs)
                ))
                fe_ys[indices] = ys[indices]
    write_toml(
        metadata | {'input_shape': xs.shape[1:]},
        path / 'metadata.toml'
    )


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_id', help='experiment ID')
    args = parser.parse_args()

    # read experiment specifications
    exp_path = Path('experiments').resolve()
    exp_spec = read_toml(exp_path / f'{args.experiment_id}.toml')

    extract_features(args.experiment_id, exp_spec)


if __name__ == '__main__':
    main()
