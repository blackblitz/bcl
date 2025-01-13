"""Training package."""

from importlib import import_module
import numpy as np

from dataops.io import read_task
from train.trainer import module_map as trainer_module_map

from evaluate import metrics


def select_trainers(exp_spec, model, mspec, results_path, ts_path, metadata):
    """
    Select one trainer among those sharing the same name.

    This function is for hyperparmeter tuning via grid search.
    """
    trainer_specs_map = {}
    for trainer_spec in exp_spec['trainers']:
        name = trainer_spec['name']
        trainer_specs_map.setdefault(name, [])
        trainer_specs_map[name].append(trainer_spec)
    for name, trainer_specs in trainer_specs_map.items():
        if len(trainer_specs) > 1:
            validation_scores = np.zeros((len(trainer_specs),))
            for i, trainer_spec in enumerate(trainer_specs):
                trainer_id = trainer_spec['id']
                trainer_class = getattr(
                    import_module(trainer_module_map[trainer_spec['name']]),
                    trainer_spec['name']
                )
                hparams = trainer_spec['hparams']['predict']
                task_id = metadata['length']
                predictor = trainer_class.predictor_class.from_checkpoint(
                    model, mspec, hparams,
                    results_path / f'ckpt/{trainer_id}_{task_id}'
                )
                metric = getattr(
                    metrics, exp_spec['evaluation']['validation_metric']
                )
                validation_scores[i] = np.mean([
                    metric(predictor, *read_task(ts_path, 'validation', i))
                    for i in range(1, task_id + 1)
                ])
            trainer_specs_map[name] = trainer_specs[validation_scores.argmax()]

        else:
            trainer_specs_map[name] = trainer_specs[0]
    return trainer_specs_map
