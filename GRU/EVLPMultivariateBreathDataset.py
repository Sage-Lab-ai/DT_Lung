import numpy as np
from torch.utils.data import Dataset


class EVLPMultivariateBreathDataset(Dataset):
    def __init__(self, selected_cases, X, Y, static, static_norm=None):
        self.selected_cases = selected_cases
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.static = static.astype(np.float32) if static is not None else np.empty((len(X), 0), dtype=np.float32)
        self.static_norm = static_norm
        assert self.X.shape[0] == self.Y.shape[0]
        assert self.X.shape[0] == self.static.shape[0] if self.static is not None else True

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if self.static is None:
            return (self.X[idx], None), self.Y[idx]
        return (self.X[idx], self.static[idx]), self.Y[idx]
        
