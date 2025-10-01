from functools import reduce
import numpy as np
import torch
import itertools

class XOR():
    def xor(self, x):
        return reduce(lambda a, b: a ^ b, x)
    
    def get_dataset(self, in_features = 2, unique = True, split_percent = 0.8):
        X = np.array(list(itertools.product([0, 1], repeat=in_features)))
        y = np.array([self.xor(x) for x in X])

        X = torch.tensor(X).float()
        y = torch.tensor(y).long()

        if unique:
            split_size = max(int(len(X)*split_percent), 1)

            n_samples = X.shape[0]
            indices = np.random.permutation(n_samples)
    
            train_idx = indices[:split_size]
            val_idx = indices[split_size:]
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            return X_train, y_train, X_val, y_val
        else:
            return X, y, X, y
        