"""Preparation of task sequences."""

seed = 1337

medmnist_src = 'data/medmnist'
pytorch_src = 'data/pytorch'
sd198_src = 'data/sd198'

singleton = [
    'cifar10', 'cifar100', 'emnist_letters', 'fashionmnist',
    'iris', 'mnist', 'sd169', 'svhn', 'wine'
]
split = [
    'split_cifar10', 'split_dermamnist',
    'split_iris', 'split_iris_2',
    'split_mnist', 'split_wine'
]
synthetic = ['santong', 'sinusoid', 'triangle']

module_map = (
    dict.fromkeys(singleton, 'dataprep.singleton')
    | dict.fromkeys(split, 'dataprep.split')
    | dict.fromkeys(synthetic, 'dataprep.synthetic')
)
