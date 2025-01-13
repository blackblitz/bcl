"""Training script."""

import argparse
import datetime
from importlib import import_module
import json
from pathlib import Path

from jax import random
import jax.numpy as jnp
import orbax.checkpoint as ocp
from tqdm import tqdm

from dataops.io import read_task, read_toml
from evaluate import metrics
from models import ModelSpec, module_map, NLL

from train.predictor import MAPPredictor

from train.trainer.smi.simple import Finetuning


def fetrain(exp_id, exp_spec):
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
    print(
        model.tabulate(
            random.key(1337),
            jnp.zeros((1, *exp_spec['feature_extractor']['spec']['in_shape']))
        )
    )
    mspec = ModelSpec(
        nll=NLL[exp_spec['feature_extractor']['spec']['nll']],
        in_shape=exp_spec['feature_extractor']['spec']['in_shape'],
        out_shape=exp_spec['feature_extractor']['spec']['out_shape'],
        cweight=exp_spec['feature_extractor']['spec']['cweight']
    )
    trainer = Finetuning(
        model, mspec,
        exp_spec['feature_extractor']['hparams']['train']
    )
    for epoch_num, state in enumerate(tqdm(
        trainer.train(train_xs, train_ys),
        total=exp_spec['feature_extractor']['hparams']['train']['n_epochs'],
        leave=False, unit='epoch'
    ), start=1):
        pass
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
    ckpt_path = Path('results').resolve() / 'ckpt' / 'feature_extractor'
    ocp.test_utils.erase_and_create_empty(ckpt_path)
    ckpt_path.rmdir()
    with ocp.StandardCheckpointer() as ckpter:
        ckpter.save(ckpt_path, trainer.state.params)


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_id', help='experiment ID')
    args = parser.parse_args()

    # read experiment specifications
    exp_path = Path('experiments').resolve()
    exp_spec = read_toml(exp_path / f'{args.experiment_id}.toml')

    fetrain(args.experiment_id, exp_spec)


if __name__ == '__main__':
    main()
