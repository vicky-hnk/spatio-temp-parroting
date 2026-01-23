"""Utility functions for training and evaluating model."""
import random
import os
import json

import torch
import numpy as np


def load_config(config_path="config_local.json"):
    """
    Load a YAML config file and merge default settings with
    environment-specific settings.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def set_seeds(seed=0, deterministic=False, tag_mlflow=False):

    # Python & Libs
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch & Cuda
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if tag_mlflow:
        try:
            import mlflow
            mlflow.set_tag("seed", int(seed))
            mlflow.set_tag("deterministic", bool(deterministic))
        except ImportError:
            pass
    return seed



class EarlyStopping:
    def __init__(self, patience=10, delta=0.00001, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Metrics:
    """Calculates prediction errors:
    # step_error: per-step, flatten B,N,C for each T separately
    # seq_error: for the whole sequence
    """
    def __init__(self, metric_flags=None):
        self.metric_flags = metric_flags

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def mse(values_pred, values_true):
        squared_error = (values_pred - values_true) ** 2
        step_errors = squared_error.mean(dim=(0, 2))
        seq_errors = squared_error.mean()
        return seq_errors, step_errors

    @staticmethod
    def mae(values_pred, values_true):
        abs_error = torch.abs(values_pred - values_true)
        step_errors = abs_error.mean(dim=(0, 2))
        seq_errors = abs_error.mean()
        return seq_errors, step_errors

    @staticmethod
    def masked_mae(values_pred, values_true, null_val=0.0):
        mask = (values_true != null_val).float()
        abs_error = torch.abs(values_pred - values_true) * mask
        step_errors = abs_error.sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp_min(1)
        seq_errors = abs_error.sum() / mask.sum().clamp_min(1)
        return seq_errors, step_errors

    @staticmethod
    def masked_rmse(values_pred, values_true, null_val=0.0):
        mask = (values_true != null_val).float()
        squared_error = ((values_pred - values_true) ** 2) * mask
        step_mse = squared_error.sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp_min(1)
        seq_mse = squared_error.sum() / mask.sum().clamp_min(1)
        return torch.sqrt(seq_mse), torch.sqrt(step_mse)

    @staticmethod
    def masked_mape(values_pred, values_true, null_val=0.0, eps=1e-4):
        mask = (values_true != null_val).float()
        # Further mask out tiny true values to avoid division by zero or huge errors
        mask *= (values_true.abs() > eps).float()

        denominator = values_true.abs().clamp_min(eps)
        abs_percent_err = (torch.abs(values_pred - values_true) / denominator) * mask

        step_errors = (abs_percent_err.sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp_min(1)) * 100
        seq_errors = (abs_percent_err.sum() / mask.sum().clamp_min(1)) * 100
        return seq_errors, step_errors

    def calculate_metrics(self, values_pred, values_true):
        out = {}
        for flag in self.metric_flags:
            func = getattr(self, flag, None)
            if callable(func):
                # We expect the function to return (sequence_error, step_errors)
                seq_err, step_err = func(values_pred, values_true)
                out[flag.upper()] = seq_err
                out[f'{flag.upper()}_HORIZON'] = step_err
        return out

