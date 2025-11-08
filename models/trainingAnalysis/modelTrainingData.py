from dataclasses import dataclass, field
from typing import Dict, List

from models.trainingAnalysis.epochData import EpochData



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
    
    def get_epoch_numbers(self) -> List[int]:
        return sorted(self.epochs.keys())
