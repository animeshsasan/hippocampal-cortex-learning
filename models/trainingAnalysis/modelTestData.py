from dataclasses import dataclass, field
from typing import List

import numpy as np

from metrics.bootstrapUtils import get_bootstrapped_value


@dataclass
class ModelTestData:
    """Represents test data for a single model"""
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    
    def add_test_data(self, train_acc: float, val_acc: float):
        """Add test data"""
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
    
    def get_train_mean_std(self):
        """Get mean and std of training accuracies"""
        return np.mean(self.train_acc), np.std(self.train_acc)

    def get_val_mean_std(self):
        """Get mean and std of training accuracies"""
        return np.mean(self.val_acc), np.std(self.val_acc)

    def get_val_mean_bootstrapped(self):
        bt_l, bt_u = get_bootstrapped_value(self.val_acc)
        return np.mean(self.val_acc), bt_l, bt_u

    def get_train_mean_bootstrapped(self):
        bt_l, bt_u = get_bootstrapped_value(self.train_acc)
        return np.mean(self.train_acc), bt_l, bt_u