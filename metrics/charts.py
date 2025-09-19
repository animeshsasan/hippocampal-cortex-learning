import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

TEST = "test"
TRAIN = "train"


class TrainingCharts(Enum):
    LOSS = "loss_while_training"
    ACC = "acc_while_training"

class TestCharts(Enum):
    TRAIN_ACC = "train_acc"
    VAL_ACC = "val_acc"

class ChartUtil():
    def __init__(self,):
        self.data = {}

    def line_chart(self, X, Y, xlabel, ylabel, title="Line Chart"):
        plt.plot(X, Y)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        plt.show()
    
    def add_train_data(self, model, loss_values, accuracy_over_epochs, epoch_numbers):
        if TRAIN in self.data:
            train_data = self.data[TRAIN]
        else:
            train_data = {}

        if model not in train_data:
            train_data[model] = {}

        n_values = 0
        for epoch in epoch_numbers:
            if epoch not in train_data[model]:
                train_data[model][epoch] = {}
                train_data[model][epoch][TrainingCharts.LOSS] = []
                train_data[model][epoch][TrainingCharts.ACC] = []
            
            train_data[model][epoch][TrainingCharts.LOSS].append(loss_values[n_values])
            train_data[model][epoch][TrainingCharts.ACC].append(accuracy_over_epochs[n_values])
            n_values += 1

        self.data[TRAIN] = train_data


    def plot_training_data_for(self, value_to_plot = "loss", models = None, run = None):
        assert TRAIN in self.data
        if value_to_plot == "loss":
            value_to_plot = TrainingCharts.LOSS
        else:
            value_to_plot  = TrainingCharts.ACC
        for model in self.data[TRAIN]:
            if models is not None and model not in models:
                continue
            train_data = self.data[TRAIN][model]

            if run is None:
                epochs = []
                mean_acc = []
                std_acc = []
                for epoch in train_data:
                    epochs.append(epoch)
                    mean_acc.append(np.mean(train_data[epoch][value_to_plot]))
                    std_acc.append(np.std(train_data[epoch][value_to_plot]))

                plt.errorbar(epochs, mean_acc, std_acc, label=model)
            else:
                epochs = []
                acc = []
                for epoch in train_data:
                    epochs.append(epoch)
                    acc.append(train_data[epoch][value_to_plot][run])
                
                plt.plot(epochs, acc, label=model)
        
        plt.xlabel("Epoch Number")

        plt.legend()
        if value_to_plot == TrainingCharts.LOSS:
            title = "Loss vs Epoch during training"
            plt.ylabel("Loss")
        else:
            title = "Accuracy vs Epoch during training"
            plt.ylabel("Accuracy")
        plt.title(title)
        
    def add_test_data(self, model, train_accuracy, val_accuracy):
        if TEST in self.data:
            test_data = self.data[TEST]
        else:
            test_data = {}

        if model not in test_data:
            test_data[model] = {}
            if TestCharts.TRAIN_ACC not in test_data[model]:
                test_data[model][TestCharts.TRAIN_ACC] = []
            if TestCharts.VAL_ACC not in test_data[model]:
                test_data[model][TestCharts.VAL_ACC] = []

        test_data[model][TestCharts.TRAIN_ACC].append(train_accuracy) 
        test_data[model][TestCharts.VAL_ACC].append(val_accuracy)
        self.data[TEST] = test_data

    def plot_test_accu_for_models(self, models = None):
        assert TEST in self.data

        model_name = []
        tr_mean = []
        t_mean = []
        tr_variance = []
        t_variance = []

        for model in self.data[TEST]:
            if models is not None and model not in models:
                continue
            test_data = self.data[TEST][model]
            model_name.append(model)
            tr_mean.append(np.mean(test_data[TestCharts.TRAIN_ACC]))
            t_mean.append(np.mean(test_data[TestCharts.VAL_ACC]))
            tr_variance.append(np.std(test_data[TestCharts.TRAIN_ACC]))    
            t_variance.append(np.std(test_data[TestCharts.VAL_ACC]))
        plt.errorbar(model_name, tr_mean, tr_variance, color='r', fmt ='o', label="Train Accuracy")    
        plt.errorbar(model_name, t_mean, t_variance, color='g', fmt='o', label="Val Accuracy")    
        plt.xlabel("Model")
        plt.ylabel("Accuracy")

        plt.legend()
        plt.title("Post-training accuracy")
