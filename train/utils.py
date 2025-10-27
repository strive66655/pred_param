import numpy as np


def pct_rmse(pred, true):
    mse = ((pred - true)**2).mean(axis=1)
    rmse = np.sqrt(mse)
    den = np.mean(np.abs(true), axis=1) + 1e-8
    pct = (rmse / den) * 100.0
    return float(np.mean(pct))


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-8):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0
        self.early_stop = False

    def step(self, value):
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
