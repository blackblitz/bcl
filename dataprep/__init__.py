"""Preparation of task sequences."""

seed = 1337

isic = ['bcn12', 'cisplitham8', 'displitham6']
pytorch = [
    'cifar100',
    'cisplitcifar10', 'cisplitmnist',
    'displitcifar8', 'displitmnist',
    'emnistletters'
]
sklearn = ['cisplitiris', 'cisplitiris2', 'cisplitwine']
synthetic = ['sinusoid', 'triangle']

module_map = (
    dict.fromkeys(isic, 'dataprep.isic')
    | dict.fromkeys(pytorch, 'dataprep.pytorch')
    | dict.fromkeys(sklearn, 'dataprep.sklearn')
    | dict.fromkeys(synthetic, 'dataprep.synthetic')
)
