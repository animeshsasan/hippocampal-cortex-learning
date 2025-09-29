from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import uuid

TEST = "test"
TRAIN = "train"


class TrainingCharts(Enum):
    LOSS = "loss_while_training"
    ACC = "acc_while_training"

class TestCharts(Enum):
    TRAIN_ACC = "train_acc"
    VAL_ACC = "val_acc"

class ChartUtil():
    def __init__(self, data = None):
        if data is not None:
            self.data = data
        else:
            self.data = {}

    def get_plot(self, models = None, height_alpha = 0.3, width_alpha = 0.3):
        if models is not None:
            n_models = len(models)
        else:
            n_models = len(self.get_models())
    
        # Scale figure size based on number of legend entries
        base_width, base_height = 8, 6
        extra_height = min(max(0, (n_models - 5) * height_alpha), 50)  # grow with many models
        extra_width = min(max(0, (n_models - 5) * width_alpha), 100)  # grow with many models
        
        return plt.subplots(figsize=(base_width + extra_width, base_height + extra_height))
    
    def get_models(self):
        if TRAIN in self.data:
            return list(self.data[TRAIN].keys())
        elif TEST in self.data:
            return list(self.data[TEST].keys())
        else:
            return []
    
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


    def plot_training_data_for(self, value_to_plot = "loss", models = None, run = None, width_alpha = 0.3, height_alpha = 0.3):
        assert TRAIN in self.data
        if value_to_plot == "loss":
            value_to_plot = TrainingCharts.LOSS
        else:
            value_to_plot  = TrainingCharts.ACC

        fig, ax = self.get_plot(models, width_alpha = width_alpha, height_alpha = height_alpha)

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

                ax.errorbar(epochs, mean_acc, std_acc, label=model)
            else:
                epochs = []
                acc = []
                for epoch in train_data:
                    epochs.append(epoch)
                    acc.append(train_data[epoch][value_to_plot][run])
                
                ax.plot(epochs, acc, label=model)
        
        ax.set_xlabel("Epoch Number")

        ax.legend()
        if value_to_plot == TrainingCharts.LOSS:
            title = "Loss vs Epoch during training"
            ax.set_ylabel("Loss")
        else:
            title = "Accuracy vs Epoch during training"
            ax.set_ylabel("Accuracy")
        ax.set_title(title)
        return fig
        
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

    def plot_test_accu_for_models(self, models = None, width_alpha = 1):
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

        fig, ax = self.get_plot(models, width_alpha=width_alpha)

        ax.errorbar(model_name, tr_mean, tr_variance, color='r', fmt='o', label="Train Accuracy")    
        ax.errorbar(model_name, t_mean, t_variance, color='g', fmt='o', label="Val Accuracy")    
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_title("Post-training accuracy")

        return fig
    
    def combine_charts(self, charts: ChartUtil, suffix = ""):
        combined_data: dict[str, dict] = {}
        combined_data[TRAIN] = {}
        combined_data[TEST] = {}
        datasets = [TRAIN, TEST]

        for dataset in datasets:
            for key in self.data[dataset]:
                combined_data[dataset][key] = self.data[dataset][key]

        for dataset in datasets:
            for model in charts.data[dataset]:

                new_model_key = model + suffix
                if new_model_key in combined_data.keys():
                    new_model_key += " " + str(uuid.uuid4())[:4]

                combined_data[dataset][new_model_key] = charts.data[dataset][model]

        new_chart = ChartUtil(combined_data)
        return new_chart
    
    def add_model_suffix(self, suffix = ""):
        assert suffix != ""

        datasets = [TRAIN, TEST]

        for dataset in datasets:
            for key in self.data[dataset]:
                self.data[dataset][key + suffix] = self.data[dataset][key]
                self.data[dataset].pop(key)

