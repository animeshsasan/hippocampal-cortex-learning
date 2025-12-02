from dataclasses import dataclass, field
from typing import List

import numpy as np

from metrics.bootstrapUtils import get_bootstrapped_value
from models.enums.chartType import TrainingCharts
import scipy.stats as st # type: ignore[import]


@dataclass
class EpochData:
    """Represents training data for a single epoch"""
    loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)

    def get_value_mean_std_by_type(self, chart_type: TrainingCharts, needCI = False):
        values = self.get_values_by_type(chart_type)
        if needCI:
            sem = st.sem(values, nan_policy='omit')
            range = 1.96 * sem
        else:
            range = np.std(values)

        return np.mean(values), range
    
    def get_value_mean_bootstrap_by_type(self, chart_type: TrainingCharts):
        values = self.get_values_by_type(chart_type)
        bt_l, bt_u = get_bootstrapped_value(values)
        return np.mean(values), bt_l, bt_u
    
    def get_values_by_type(self, chart_type: TrainingCharts):
        if chart_type == TrainingCharts.LOSS:
            values = self.loss
        elif chart_type == TrainingCharts.TRAIN_ACC:
            values = self.train_acc
        elif chart_type == TrainingCharts.VAL_ACC:
            values = self.val_acc
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")

        return values
    
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