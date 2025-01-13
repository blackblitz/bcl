"""Pre-training a feature extractor."""

import datetime
from importlib import import_module
import json
from pathlib import Path

from jax import random
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm

from dataops.array import batch, get_pass_size
from dataops.io import get_filenames, read_task, read_toml, write_toml
from evaluate import metrics
from models import ModelSpec, module_map, NLL
from train.predictor import MAPPredictor
from train.training import init
from train.trainer.smi.simple import Finetuning


def extract_features(exp_id, exp_spec):
    """Extract features."""
    # read metadata
    ts_path = (
        Path('data/prepped').resolve() / exp_spec['task_sequence']['name']
    )
    metadata = read_toml(ts_path / 'metadata.toml')

    # prepare directory for checkpoints
    result_path = Path('results').resolve() / exp_id
    ckpt_path = result_path / 'ckpt'

    # load checkpoint
    model = getattr(
        import_module(module_map[exp_spec['feature_extractor']['name']]),
        exp_spec['feature_extractor']['name']
    )(**exp_spec['feature_extractor']['args'])
    mspec = ModelSpec(
        nll=NLL[exp_spec['feature_extractor']['spec']['nll']],
        in_shape=exp_spec['feature_extractor']['spec']['in_shape'],
        out_shape=exp_spec['feature_extractor']['spec']['out_shape'],
        cratio=exp_spec['feature_extractor']['spec']['cratio'],
        cscale=exp_spec['feature_extractor']['spec']['cscale']
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
    splits = tqdm(['training', 'validation', 'testing'], unit='split')
    for split in splits:
        splits.set_description(f'Extracting features for {split}')
        task_ids = tqdm(
            range(1, metadata['length'] + 1), leave=False, unit='task'
        )
        for task_id in task_ids:
            task_ids.set_description(f'Task {task_id}')
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


def train(exp_id, exp_spec):
    """Train the feature extractor and save checkpoint to path."""
    # train and log
    Path('logs').mkdir(exist_ok=True)
    log_path = (
        Path('logs')
        / f'{exp_id}_feature_extractor_'
        f'{datetime.datetime.now().isoformat()}.jsonl'
    )
    ts_path = (
        Path('data/prepped').resolve()
        / exp_spec['feature_extractor']['task_sequence_name']
    )
    train_xs, train_ys = read_task(ts_path, 'training', 1)
    val_xs, val_ys = read_task(ts_path, 'validation', 1)
    model = getattr(
        import_module(module_map[exp_spec['feature_extractor']['name']]),
        exp_spec['feature_extractor']['name']
    )(**exp_spec['feature_extractor']['args'])
    # print(
    #     model.tabulate(
    #         random.key(1337),
    #         jnp.zeros((1, *exp_spec['feature_extractor']['spec']['in_shape']))
    #     )
    # )
    mspec = ModelSpec(
        nll=NLL[exp_spec['feature_extractor']['spec']['nll']],
        in_shape=exp_spec['feature_extractor']['spec']['in_shape'],
        out_shape=exp_spec['feature_extractor']['spec']['out_shape'],
        cratio=exp_spec['feature_extractor']['spec']['cratio'],
        cscale=exp_spec['feature_extractor']['spec']['cscale']
    )
    trainer = Finetuning(
        model, mspec,
        exp_spec['feature_extractor']['hparams']['train']
    )
    states = tqdm(
        trainer.train(train_xs, train_ys),
        total=exp_spec['feature_extractor']['hparams']['train']['n_epochs'],
        leave=False, unit='epoch'
    )
    for epoch_num, state in enumerate(states, start=1):
        states.set_description(
            f'Training feature extractor - Epoch {epoch_num}'
        )
        predictor = MAPPredictor(
            model,
            mspec,
            exp_spec['feature_extractor']['hparams']['predict'],
            state.params
        )
        result = {'epoch_num': epoch_num}
        for metric in exp_spec['evaluation']['metrics']:
            result[metric] = getattr(metrics, metric)(
                predictor, val_xs, val_ys
            )
        with open(log_path, mode='a') as file:
            print(json.dumps(result), file=file)

    # save checkpoint
    ckpt_path = (
        Path('results').resolve() / exp_id / 'ckpt' / 'feature_extractor'
    )
    ckpt_path.mkdir(parents=True, exist_ok=True)
    ocp.test_utils.erase_and_create_empty(ckpt_path)
    ckpt_path.rmdir()
    with ocp.StandardCheckpointer() as ckpter:
        ckpter.save(ckpt_path, trainer.state.params)
