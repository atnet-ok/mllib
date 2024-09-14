from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
import numpy as np
import torch
from torch import nn
import torchvision
import pandas as pd


def get_metrics(task):
    if task == "classification":
        criterion = nn.BCEWithLogitsLoss()
        get_score = get_classification_score
        is_best_model = is_highest_accuracy

    elif task == "generation":
        criterion = nn.MSELoss()
        get_score = None  # To be implemented
        is_best_model = None  # To be implemented

    elif task == "regression":
        criterion = nn.MSELoss()
        get_score = None  # To be implemented
        is_best_model = None  # To be implemented

    elif task == "semaseg":
        criterion = nn.BCELoss()
        get_score = None  # To be implemented
        is_best_model = None  # To be implemented

    return criterion, get_score, is_best_model


def get_regression_score(y_pred, y_true):
    return 0


def get_generation_score(y_pred, y_true):
    return 0


def get_semaseg_score(y_pred, y_true):
    return 0


def get_classification_score(y_pred, y_true):
    cls_true = [np.argmax(i) for i in y_true]
    cls_pred = [np.argmax(i) for i in y_pred]

    accuracy = accuracy_score(cls_true, cls_pred)

    return accuracy


def is_highest_accuracy(best_score, score):
    if best_score is None:
        best_score = 0
    return True if best_score < score else False
