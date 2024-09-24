"""Training script."""

import argparse
from pathlib import Path

import orbax.checkpoint as ocp
from tqdm import tqdm

from dataops.io import read_task, read_toml
import models
import train


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

    # pre-train and save checkpoint
    fe_model = getattr(
        models, exp['feature_extractor']['name']
    )(**exp['feature_extractor']['args'])
    fe_model_spec = models.ModelSpec(
        nll=models.NLL[exp['feature_extractor']['spec']['nll']],
        in_shape=exp['feature_extractor']['spec']['in_shape'],
        out_shape=exp['feature_extractor']['spec']['out_shape']
    )
    fe_trainer = train.Finetuning(
        fe_model, fe_model_spec, exp['feature_extractor']['immutables']
    )
    fe_ts_path = (
        Path('data').resolve()
        / exp['feature_extractor']['task_sequence_name']
    )
    for fe_state in tqdm(
        fe_trainer.train(*read_task(fe_ts_path, 'training', 1)),
        total=exp['feature_extractor']['immutables']['n_epochs'],
        leave=False, unit='epoch'
    ):
        pass
    path = ckpt_path / 'feature_extractor'
    ocp.test_utils.erase_and_create_empty(path)
    path.rmdir()
    with ocp.StandardCheckpointer() as ckpter:
        ckpter.save(path, fe_state.params)


if __name__ == '__main__':
    main()
