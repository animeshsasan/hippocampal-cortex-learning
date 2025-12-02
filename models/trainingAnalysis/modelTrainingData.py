from dataclasses import dataclass, field
from typing import Dict, List

from metrics.bootstrapUtils import get_bootstrapped_value
from models.enums.chartType import TrainingCharts
from models.trainingAnalysis.epochData import EpochData
import numpy as np


@dataclass
class ModelTrainingData:
    """Represents all training data for a single model"""
    epochs: Dict[int, EpochData] = field(default_factory=dict)
    
    def get_or_create_epoch(self, epoch: int) -> EpochData:
        """Get existing epoch data or create new one"""
        if epoch not in self.epochs:
            self.epochs[epoch] = EpochData()
        return self.epochs[epoch]
    
    def add_epoch_data(self, epoch: int, loss: float, train_acc: float, val_acc: float):
        """Add data for a specific epoch"""
        epoch_data = self.get_or_create_epoch(epoch)
        epoch_data.loss.append(loss)
        epoch_data.train_acc.append(train_acc)
        epoch_data.val_acc.append(val_acc)

    def add_multiple_epoch_data(self, epochs: list[int], loss_values: list, train_acc_values: list, val_acc_values: list):
        """Add multiple data points for a specific epoch"""
        n_values = 0
        for epoch in epochs:
            self.add_epoch_data(
                epoch=epoch,
                loss=loss_values[n_values],
                train_acc=train_acc_values[n_values],
                val_acc=val_acc_values[n_values]
            )
            n_values += 1

    def get_epochs_to_thr(self, modelName: str, thresholds: list[float] = [90., 95.]):
        epochs = list(self.epochs.keys())
        num_runs = len(self.epochs[0].train_acc)
        
        epochs_to_threshold: dict[float, list[int]] = {thr: [] for thr in thresholds}
        
        for run_idx in range(num_runs):
            # extract the accuracy curve for this run
            acc_curve = [self.epochs[epoch].train_acc[run_idx] for epoch in epochs]

            for thr in thresholds:
                # find first epoch reaching threshold
                reached = [ep for ep, acc in zip(epochs, acc_curve) if acc >= thr]

                if reached:
                    epochs_to_threshold[thr].append(reached[0])
            
            if not reached:
                print(f"Error computing epochs to threshhold for {modelName}: Does not reach thr {thr} in run {run_idx}")
        
        return epochs_to_threshold
    
    def get_epochs_to_thr_mean_bt(self, modelName: str, thresholds: list[float] = [90., 95.]):
    
        epochs_to_threshold = self.get_epochs_to_thr(modelName, thresholds)
        mean: dict[float, float] = {}
        bt_l: dict[float, float] = {}
        bt_h: dict[float, float] = {}
        for thr in thresholds:
            mean[thr] = np.mean(epochs_to_threshold[thr])
            bt_l[thr], bt_h[thr] = get_bootstrapped_value(epochs_to_threshold[thr])
        
        return mean, bt_l, bt_h
    
    def get_mean_bt_acc(self,):
        mean_acc = []
        bt_ls = []
        bt_us = []
        epochs = list(self.epochs.keys())

        for epoch in self.epochs:
            epoch_data = self.epochs[epoch]

            mean, bt_l, bt_u = epoch_data.get_value_mean_bootstrap_by_type(
                TrainingCharts.TRAIN_ACC
            )
            mean_acc.append(mean)
            bt_ls.append(bt_l)
            bt_us.append(bt_u)
        
        return mean_acc, bt_ls, bt_us, epochs
    
    def get_epoch_numbers(self) -> List[int]:
        return sorted(self.epochs.keys())
