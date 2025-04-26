"""Bayesian continual learning."""

import datetime
import json
from importlib import import_module
from pathlib import Path

import orbax.checkpoint as ocp

from bcl.dataops.io import read_toml
from bcl.feature_extractor import extract_features, train
from bcl.models import ModelSpec, NLL
from bcl.models import module_map as models_module_map
from bcl.train.trainer import coreset_memmap_path, coreset_zarr_path


class Experiment:
    """Experiment."""

    def __init__(self, experiment_id):
        """Initialize self."""
        self.id = experiment_id
        self.paths = {
            'checkpoint': Path('results').resolve() / self.id / 'ckpt',
            'data': None,
            'log': (
                Path('logs') / f'{self.id}_'
                f'{datetime.datetime.now().isoformat()}.jsonl'
            ),
            'memmap': coreset_memmap_path,
            'result': Path('results').resolve() / self.id,
            'zarr': coreset_zarr_path
        }
        self.metadata = None
        self.model = None
        self.mspec = None
        self.spec = None

    def make_cpath(self):
        """Make checkpoint path."""
        self.paths['checkpoint'].mkdir(parents=True, exist_ok=True)

    def read_spec(self):
        """Read specifications."""
        self.spec = read_toml(
            Path('experiments').resolve() / f'{self.id}.toml'
        )

    def make_data(self, is_eval=False):
        """Make data."""
        if 'feature_extractor' in self.spec:
            if not is_eval:
                train(self.id, self.spec)
            extract_features(self.id, self.spec)
            self.paths['data'] = Path('data/features').resolve()
        else:
            self.paths['data'] = (
                Path('data/prepped').resolve()
                / self.spec['task_sequence']['name']
            )

    def read_metadata(self):
        """Read metadata."""
        self.metadata = read_toml(self.paths['data'] / 'metadata.toml')

    def make_model(self):
        """Make model."""
        self.model = getattr(
            import_module(models_module_map[self.spec['model']['name']]),
            self.spec['model']['name']
        )(**self.spec['model']['args'])
        self.mspec = ModelSpec(
            nll=NLL[self.spec['model']['spec']['nll']],
            in_shape=self.spec['model']['spec']['in_shape'],
            out_shape=self.spec['model']['spec']['out_shape'],
            cratio=self.spec['model']['spec']['cratio'],
            cscale=self.spec['model']['spec']['cscale']
        )

    def setup(self, is_eval=False):
        """Set up experiment."""
        self.make_cpath()
        self.read_spec()
        self.make_data(is_eval=is_eval)
        self.read_metadata()
        self.make_model()

    def index_cpath(self, trainer_id, task_id):
        """Index checkpoint path."""
        path = self.paths['checkpoint'] / f'{trainer_id}_{task_id}'
        path.mkdir(parents=True, exist_ok=True)
        ocp.test_utils.erase_and_create_empty(path)
        path.rmdir()
        return path

    def log(self, result):
        """Log."""
        with open(self.paths['log'], mode='a') as file:
            print(json.dumps(result), file=file)

    def teardown(self):
        """Tear down experiment."""
        ocp.test_utils.erase_and_create_empty(self.paths['memmap'])
        self.paths['memmap'].rmdir()
        ocp.test_utils.erase_and_create_empty(self.paths['zarr'])
        self.paths['zarr'].rmdir()
