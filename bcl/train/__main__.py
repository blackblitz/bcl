"""Training script."""

import argparse
from importlib import import_module

import orbax.checkpoint as ocp
from tqdm import tqdm

from .. import Experiment
from ..dataops.io import read_task
from ..evaluate import metrics
from ..train.trainer import module_map as trainer_module_map


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', help='experiment ID')
    args = parser.parse_args()

    exp = Experiment(args.exp_id)
    exp.setup()
    with ocp.StandardCheckpointer() as ckpter:
        trainer_specs = tqdm(exp.spec['trainers'], leave=False, unit='trainer')
        for trainer_spec in trainer_specs:
            trainer_specs.set_description(
                f'Training for {trainer_spec["id"]}'
            )
            trainer_id = trainer_spec['id']
            trainer_class = getattr(
                import_module(trainer_module_map[trainer_spec['name']]),
                trainer_spec['name']
            )
            train_hparams = trainer_spec['hparams']['train']
            predict_hparams = trainer_spec['hparams']['predict']

            trainer = trainer_class(exp.model, exp.mspec, train_hparams)
            task_ids = tqdm(
                range(1, exp.metadata['length'] + 1),
                leave=False, unit='task'
            )
            for task_id in task_ids:
                task_ids.set_description(f'Task {task_id}')
                xs, ys = read_task(exp.paths['data'], 'training', task_id)
                states = tqdm(
                    trainer.train(xs, ys),
                    total=train_hparams['n_epochs'],
                    leave=False, unit='epoch'
                )
                for epoch_num, state in enumerate(states, start=1):
                    states.set_description(f'Epoch {epoch_num}')
                    predictor = trainer.predictor_class(
                        exp.model, exp.mspec, predict_hparams, state.params
                    )
                    result = {
                        'trainer_id': trainer_id,
                        'task_id': task_id,
                        'epoch_num': epoch_num
                    }
                    for metric in exp.spec['evaluation']['metrics']:
                        result[metric] = [
                            getattr(metrics, metric)(
                                predictor,
                                *read_task(exp.paths['data'], 'validation', i)
                            ) for i in range(1, exp.metadata['length'] + 1)
                        ]
                    exp.log(result)
                    # with open(log_path, mode='a') as file:
                    #     print(json.dumps(result), file=file)
                # path = ckpt_path / f'{trainer_id}_{task_id}'
                # path.mkdir(parents=True, exist_ok=True)
                # ocp.test_utils.erase_and_create_empty(path)
                # path.rmdir()
                ckpter.save(
                    exp.index_cpath(trainer_id, task_id),
                    trainer.state.params
                )


if __name__ == '__main__':
    main()
