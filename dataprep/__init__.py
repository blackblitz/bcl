"""Preparation of task sequences."""

singleton = [
    'cifar10', 'cifar100', 'emnist_letters', 'fashionmnist',
    'iris', 'mnist', 'svhn', 'wine'
]
split = [
    'split_cifar10', 'split_iris', 'split_iris_2',
    'split_mnist', 'split_wine'
]
synthetic = ['santong', 'sinusoid', 'triangle']

module_map = (
    dict.fromkeys(singleton, 'dataprep.singleton')
    | dict.fromkeys(split, 'dataprep.split')
    | dict.fromkeys(synthetic, 'dataprep.synthetic')
)
