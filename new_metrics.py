# Source: https://github.com/miguelgfierro/codebase/blob/master/python/machine_learning/metrics.py
# Source: https://github.com/Azure/fast_retraining/blob/master/experiments/05_FraudDetection.ipynb

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, \
    recall_score, f1_score


def evaluate_metrics(y_true, y_pred, metrics):
    res = {}
    for metric_name, metric in metrics.items():
        res[metric_name] = metric(y_true, y_pred)
    return res


def classification_metrics_binary_prob(y_true, y_prob, threshold=0.5):
    y_pred = np.where(y_prob > threshold, 1, 0)
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'AUC': lambda y_true, y_pred: roc_auc_score(y_true, y_prob),
        'F1': f1_score,
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def classification_metrics_multilabel(y_true, y_pred, labels):
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred,
                                                            labels,
                                                            average='weighted'),
        'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, labels,
                                                      average='weighted'),
        'F1': lambda y_true, y_pred: f1_score(y_true, y_pred, labels,
                                              average='weighted'),
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def classification_metrics_average(y_true, y_pred, average):
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred,
                                                            average=average),
        'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred,
                                                      average=average),
        'F1': lambda y_true, y_pred: f1_score(y_true, y_pred, average=average),
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def classification_metrics(y_true, y_pred):
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'AUC': roc_auc_score,
        'F1': f1_score,
    }
    return evaluate_metrics(y_true, y_pred, metrics)
