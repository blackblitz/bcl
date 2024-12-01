"""Metrics."""

import numpy as np
from sklearn.metrics import roc_auc_score

from dataops.array import batch, get_pass_size
from models import NLL


def accuracy(predictor, xs, ys):
    """Compute the accuracy."""
    pass_size = get_pass_size(xs[0].shape)
    correct = 0
    for indices in batch(pass_size, np.arange(len(ys))):
        correct += (np.asarray(predictor(xs[indices])) == ys[indices]).sum()
    return correct.item() / len(ys)


def ood_entropy_roc_auc(predictor, xs0, xs1):
    """Compute the ROC AUC based on entropy for OOD detection."""
    target = np.concatenate([np.zeros((len(xs0),)), np.ones((len(xs1),))])
    pass_size = get_pass_size(xs0[0].shape)
    score = np.concatenate([
        predictor.entropy(xs0[indices])
        for indices in batch(pass_size, np.arange(len(xs0)))
    ] + [
        predictor.entropy(xs1[indices])
        for indices in batch(pass_size, np.arange(len(xs1)))
    ])
    return roc_auc_score(target, score)


def ood_mi_roc_auc(predictor, xs0, xs1):
    """Compute the ROC AUC based on mutual information for OOD detection."""
    target = np.concatenate([np.zeros((len(xs0),)), np.ones((len(xs1),))])
    pass_size = get_pass_size(xs0[0].shape)
    score = np.concatenate([
        predictor.mutual_information(xs0[indices])
        for indices in batch(pass_size, np.arange(len(xs0)))
    ] + [
        predictor.mutual_information(xs1[indices])
        for indices in batch(pass_size, np.arange(len(xs1)))
    ])
    return roc_auc_score(target, score)


def ece(predictor, xs, ys, n_bins=100):
    """Compute the expected calibration error."""
    pass_size = get_pass_size(xs[0].shape)
    proba = np.concatenate([
        np.asarray(predictor(xs[indices], decide=False))
        for indices in batch(pass_size, np.arange(len(ys)))
    ])
    match predictor.mspec.nll:
        case NLL.SIGMOID_CROSS_ENTROPY:
            pred = proba >= 0.5
            pred_proba = np.maximum(proba, 1 - proba)
        case NLL.SOFTMAX_CROSS_ENTROPY:
            pred = proba.argmax(axis=-1)
            pred_proba = proba.max(axis=-1)
    count = np.zeros((n_bins,))
    accuracy = np.full_like(count, np.nan)
    avg_pred_proba = np.full_like(count, np.nan)
    edges = np.linspace(0., 1., num=n_bins + 1)
    for i in range(n_bins):
        mask = (edges[i] <= pred_proba) & (pred_proba < edges[i + 1])
        count[i] = mask.sum()
        if count[i] > 0:
            accuracy[i] = (pred[mask] == ys[mask]).mean()
            avg_pred_proba[i] = pred_proba[mask].mean()
    mask = count > 0
    return (
        np.abs(accuracy[mask] - avg_pred_proba[mask])
        @ (count[mask] / len(ys))
    )
