from enum import Enum


class TrainingCharts(Enum):
    LOSS = "loss"
    TRAIN_ACC = "train_acc"
    VAL_ACC = "val_acc"


class TestCharts(Enum):
    TRAIN_ACC = "train_acc"
    VAL_ACC = "val_acc"