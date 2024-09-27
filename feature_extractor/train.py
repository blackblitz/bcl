"""Training script."""

import argparse
import datetime
from importlib import import_module
import json
from pathlib import Path

import orbax.checkpoint as ocp
from tqdm import tqdm

from dataops.io import read_task, read_toml
from evaluate import metrics
from models import ModelSpec, module_map, NLL
from train.smi.simple import Finetuning


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
    fe_model = getattr(
        import_module(module_map[exp['feature_extractor']['name']]),
        exp['feature_extractor']['name']
    )(**exp['feature_extractor']['args'])
    fe_model_spec = ModelSpec(
        nll=NLL[exp['feature_extractor']['spec']['nll']],
        in_shape=exp['feature_extractor']['spec']['in_shape'],
        out_shape=exp['feature_extractor']['spec']['out_shape']
    )
    fe_trainer = Finetuning(
        fe_model, fe_model_spec,
        exp['feature_extractor']['immutables']['train']
    )
    fe_ts_path = (
        Path('data').resolve()
        / exp['feature_extractor']['task_sequence_name']
    )
    for epoch_num, fe_state in enumerate(tqdm(
        fe_trainer.train(*read_task(fe_ts_path, 'training', 1)),
        total=exp['feature_extractor']['immutables']['train']['n_epochs'],
        leave=False, unit='epoch'
    ), start=1):
        predictor = fe_trainer.predictor_class(
            fe_model,
            fe_model_spec,
            exp['feature_extractor']['immutables']['predict'],
            fe_state.params
        )
        result = {'epoch_num': epoch_num}
        for metric in exp['evaluation']['metrics']:
            result[metric] = getattr(metrics, metric)(
                predictor, *read_task(fe_ts_path, 'validation', 1)
            )
        with open(log_path, mode='a') as file:
            print(json.dumps(result), file=file)
    path = ckpt_path / 'feature_extractor'
    ocp.test_utils.erase_and_create_empty(path)
    path.rmdir()
    with ocp.StandardCheckpointer() as ckpter:
        ckpter.save(path, fe_state.params)


if __name__ == '__main__':
    main()
