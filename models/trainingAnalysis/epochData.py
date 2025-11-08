from dataclasses import dataclass, field
from typing import List

import numpy as np

from models.enums.chartType import TrainingCharts



@dataclass
class EpochData:
    """Represents training data for a single epoch"""
    loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)

    def get_value_mean_std_by_type(self, chart_type: TrainingCharts):
        if chart_type == TrainingCharts.LOSS:
            values = self.loss
        elif chart_type == TrainingCharts.TRAIN_ACC:
            values = self.train_acc
        elif chart_type == TrainingCharts.VAL_ACC:
            values = self.val_acc
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        return np.mean(values), np.std(values)
    
    def get_value(self, chart_type: TrainingCharts, run: int):
        if chart_type == TrainingCharts.LOSS:
            values = self.loss
        elif chart_type == TrainingCharts.TRAIN_ACC:
            values = self.train_acc
        elif chart_type == TrainingCharts.VAL_ACC:
            values = self.val_acc
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        return values[run]