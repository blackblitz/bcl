"""Script that reports the metametrics."""

import argparse
from pathlib import Path

import pandas as pd

from ..dataops.io import read_toml

from . import metametrics


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('report_id', help='Report ID')
    args = parser.parse_args()

    # read report specifications
    report_path = Path('reports').resolve()
    report_spec = read_toml(report_path / f'{args.report_id}.toml')

    pd.options.display.float_format = '{:.4f}'.format

    for metric in report_spec['metrics']:
        print(metric['name'])
        print((pd.DataFrame({
            exp_id: getattr(metametrics, metric['metametric'])(
                metametrics.read_metric(exp_id, metric['basemetric'])
            )
            for exp_id in report_spec['exps']
        }) * 100).round(4))


if __name__ == '__main__':
    main()
