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

from train.predictor.stateful import MAPPredictor

from .trainer import Trainer


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_id', help='experiment ID')
    args = parser.parse_args()

    # read experiment specifications
    exp_path = Path('experiments').resolve()
    exp = read_toml(exp_path / f'{args.experiment_id}.toml')

    # prepare directory for checkpoints
    result_path = Path('results').resolve() / args.experiment_id
    ckpt_path = result_path / 'ckpt'
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # train, log and checkpoint
    Path('logs').mkdir(exist_ok=True)
    log_path = (
        Path('logs')
        / f'{args.experiment_id}_feature_extractor_'
        f'{datetime.datetime.now().isoformat()}.jsonl'
    )
    ts_path = (
        Path('data').resolve()
        / exp['feature_extractor']['task_sequence_name']
    )
    train_xs, train_ys = read_task(ts_path, 'training', 1)
    val_xs, val_ys = read_task(ts_path, 'validation', 1)
    model = getattr(
        import_module(module_map[exp['feature_extractor']['name']]),
        exp['feature_extractor']['name']
    )(**exp['feature_extractor']['args'])
    print(
        model.tabulate(
            random.key(1337),
            jnp.zeros((1, 32, 32, 3)),
            train=False
        )
    )
    model_spec = ModelSpec(
        nll=NLL[exp['feature_extractor']['spec']['nll']],
        in_shape=exp['feature_extractor']['spec']['in_shape'],
        out_shape=exp['feature_extractor']['spec']['out_shape']
    )
    trainer = Trainer(
        model, model_spec,
        exp['feature_extractor']['immutables']['train'], len(train_ys)
    )
    for epoch_num, (state, var) in enumerate(tqdm(
        trainer.train(train_xs, train_ys),
        total=exp['feature_extractor']['immutables']['train']['n_epochs'],
        leave=False, unit='epoch'
    ), start=1):
        pass
        predictor = MAPPredictor(
            model,
            model_spec,
            exp['feature_extractor']['immutables']['predict'],
            state.params,
            var
        )
        result = {'epoch_num': epoch_num}
        for metric in exp['evaluation']['metrics']:
            result[metric] = getattr(metrics, metric)(
                predictor, val_xs, val_ys
            )
        with open(log_path, mode='a') as file:
            print(json.dumps(result), file=file)
    path = ckpt_path / 'feature_extractor'
    ocp.test_utils.erase_and_create_empty(path)
    path.rmdir()
    with ocp.StandardCheckpointer() as ckpter:
        ckpter.save(path, {'params': state.params} | var)


if __name__ == '__main__':
    main()
