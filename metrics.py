# Source: https://github.com/miguelgfierro/codebase/blob/master/python/machine_learning/metrics.py
# Source: https://github.com/Azure/fast_retraining/blob/master/experiments/05_FraudDetection.ipynb

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, log_loss, recall_score, f1_score


def evaluate_metrics(y_true, y_pred, metrics):
    res = {}
    for metric_name, metric in metrics.items():
        res[metric_name] = metric(y_true, y_pred)
    return res


def classification_metrics_binary_prob(y_true, y_prob, threshold=0.5):
    y_pred = np.where(y_prob > threshold, 1, 0)
    metrics = {
        "Accuracy":  accuracy_score,
        "Precision": precision_score,
        "Recall":    recall_score,
        "Log_Loss":  lambda real, pred: log_loss(real, y_prob, eps=1e-5),
        # yes, I'm using y_prob here!
        "AUC":       lambda real, pred: roc_auc_score(real, y_prob),
        "F1":        f1_score,
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def classification_metrics_multilabel(y_true, y_pred, labels):
    metrics = {
        "Accuracy":  accuracy_score,
        "Precision": lambda real, pred: precision_score(real, pred, labels,
                                                        average="weighted"),
        "Recall":    lambda real, pred: recall_score(real, pred, labels,
                                                     average="weighted"),
        "F1":        lambda real, pred: f1_score(real, pred, labels,
                                                 average="weighted"),
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def classification_metrics_average(y_true, y_pred, avg):
    metrics = {
        "Accuracy":  accuracy_score,
        "Precision": lambda real, pred: precision_score(real, pred, average=avg),
        "Recall":    lambda real, pred: recall_score(real, pred, average=avg),
        "F1":        lambda real, pred: f1_score(real, pred, average=avg),
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def classification_metrics(y_true, y_pred):
    metrics = {
        "Accuracy":  accuracy_score,
        "Precision": precision_score,
        "Recall":    recall_score,
        "AUC":       roc_auc_score,
        "F1":        f1_score,
    }
    return evaluate_metrics(y_true, y_pred, metrics)
