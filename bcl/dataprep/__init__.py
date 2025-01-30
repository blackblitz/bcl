"""Preparation of task sequences."""

seed = 1337

isic = ['bcn12', 'cisplitham8', 'displitham6']
pytorch = [
    'cifar100',
    'cisplitcifar10', 'cisplitmnist',
    'displitcifar8', 'displitmnist',
    'emnistletters'
]
sklearn = ['cisplitiris', 'cisplit2diris', 'cisplitwine']
synthetic = ['citriangle', 'disinusoid']

module_map = (
    dict.fromkeys(isic, 'bcl.dataprep.isic')
    | dict.fromkeys(pytorch, 'bcl.dataprep.pytorch')
    | dict.fromkeys(sklearn, 'bcl.dataprep.sklearn')
    | dict.fromkeys(synthetic, 'bcl.dataprep.synthetic')
)
