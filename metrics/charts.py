from __future__ import annotations
from matplotlib import cm
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import uuid

from models.enums.chartType import TestCharts, TrainingCharts
from models.trainingAnalysis.experimentResults import ExperimentResults

TEST = "test"
TRAIN = "train"


class ChartUtil():
    def __init__(self, data = None, results: ExperimentResults | None = None):
        self.colorsMap: dict[str, Colormap] = {}
        if results is None:
            self.results = ExperimentResults()
        else:
            self.results = results
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
        return self.results.get_models()
        # if TRAIN in self.data:
        #     return list(self.data[TRAIN].keys())
        # elif TEST in self.data:
        #     return list(self.data[TEST].keys())
        # else:
        #     return []
    
    def add_train_data(self, model, loss_values, train_accuracy_over_epochs, epoch_numbers, test_accuracy_over_epochs):
        # if TRAIN in self.data:
        #     train_data = self.data[TRAIN]
        # else:
        #     train_data = {}

        # if model not in train_data:
        #     train_data[model] = {}

        # n_values = 0
        # for epoch in epoch_numbers:
        #     if epoch not in train_data[model]:
        #         train_data[model][epoch] = {}
        #         train_data[model][epoch][TrainingCharts.LOSS] = []
        #         train_data[model][epoch][TrainingCharts.TRAIN_ACC] = []
        #         train_data[model][epoch][TrainingCharts.VAL_ACC] = []
            
        #     train_data[model][epoch][TrainingCharts.LOSS].append(loss_values[n_values])
        #     train_data[model][epoch][TrainingCharts.TRAIN_ACC].append(train_accuracy_over_epochs[n_values])
        #     train_data[model][epoch][TrainingCharts.VAL_ACC].append(test_accuracy_over_epochs[n_values])
        #     n_values += 1
        # self.data[TRAIN] = train_data

        model_data = self.results.get_or_create_training_model(model)
        model_data.add_multiple_epoch_data(
            epochs=epoch_numbers,
            loss_values=loss_values,
            train_acc_values=train_accuracy_over_epochs,
            val_acc_values=test_accuracy_over_epochs
        )

    
    def set_colors_map(self):
        colors = cm.get_cmap('tab20', len(self.get_models()))
        for i, modelName in enumerate(self.get_models()):
            self.colorsMap[modelName] = colors(i)

    def get_colors_map(self, models):
        colors = cm.get_cmap('tab20', len(models))
        colorsMap = {}
        for i, modelName in enumerate(models):
            colorsMap[modelName] = colors(i)
        return colorsMap

    def plot_training_data_for(self,
                               value_to_plot = TrainingCharts.LOSS, 
                               models = None, 
                               run = None, 
                               width_alpha = 0.3, 
                               height_alpha = 0.3, 
                               no_std = False,
                               set_unique_colors = False):
        # assert TRAIN in self.data

        fig, ax = self.get_plot(models, width_alpha = width_alpha, height_alpha = height_alpha)

        if set_unique_colors:
            colorMapForModels = models if models is not None else self.get_models()
            colorsMap = self.get_colors_map(colorMapForModels)
        else:
            colorsMap = self.colorsMap
            
        for model, model_data in self.results.train_data.items():
            if models is not None and model not in models:
                continue
            epochs = model_data.get_epoch_numbers()

            if run is None:
                mean_acc = []
                std_acc = []
                for epoch in model_data.epochs:
                    epoch_data = model_data.epochs[epoch]
                    
                    mean, std = epoch_data.get_value_mean_std_by_type(value_to_plot)
                    mean_acc.append(mean)
                    std_acc.append(std)

                if no_std:
                    ax.plot(epochs, mean_acc, label=model, color = colorsMap[model])
                else:
                    ax.errorbar(epochs, mean_acc, std_acc, label=model, color = colorsMap[model])
            else:
                acc = []
                for epoch in model_data.epochs:
                    epoch_data = model_data.epochs[epoch]
                    acc_value = epoch_data.get_value(value_to_plot, run)
                    acc.append(acc_value)
                
                ax.plot(epochs, acc, label=model)

        # for model in self.data[TRAIN]:
        #     if models is not None and model not in models:
        #         continue
        #     train_data = self.data[TRAIN][model]

        #     if run is None:
        #         epochs = []
        #         mean_acc = []
        #         std_acc = []
        #         for epoch in train_data:
        #             epochs.append(epoch)
        #             mean_acc.append(np.mean(train_data[epoch][value_to_plot]))
        #             std_acc.append(np.std(train_data[epoch][value_to_plot]))

        #         if no_std:
        #             ax.plot(epochs, mean_acc, label=model, color = colorsMap[model])
        #         else:
        #             ax.errorbar(epochs, mean_acc, std_acc, label=model, color = colorsMap[model])
        #     else:
        #         epochs = []
        #         acc = []
        #         for epoch in train_data:
        #             epochs.append(epoch)
        #             acc.append(train_data[epoch][value_to_plot][run])
                
        #         ax.plot(epochs, acc, label=model)
        
        ax.set_xlabel("Epoch Number")

        ax.legend()
        if value_to_plot == TrainingCharts.LOSS:
            title = "Loss vs Epoch during training"
            ax.set_ylabel("Loss")
        elif value_to_plot == TrainingCharts.VAL_ACC:
            title = "Val Accuracy vs Epoch during training"
            ax.set_ylabel("Accuracy")
        else:
            title = "Train Accuracy vs Epoch during training"
            ax.set_ylabel("Accuracy")
        ax.set_title(title)
        return fig
        
    def add_test_data(self, model, train_accuracy, val_accuracy):
        # if TEST in self.data:
        #     test_data = self.data[TEST]
        # else:
        #     test_data = {}

        # if model not in test_data:
        #     test_data[model] = {}
        #     if TestCharts.TRAIN_ACC not in test_data[model]:
        #         test_data[model][TestCharts.TRAIN_ACC] = []
        #     if TestCharts.VAL_ACC not in test_data[model]:
        #         test_data[model][TestCharts.VAL_ACC] = []

        # test_data[model][TestCharts.TRAIN_ACC].append(train_accuracy) 
        # test_data[model][TestCharts.VAL_ACC].append(val_accuracy)
        # self.data[TEST] = test_data

        model_test_data = self.results.get_or_create_test_model(model)
        model_test_data.add_test_data(train_accuracy, val_accuracy)

    def plot_test_accu_for_models(self, models = None, width_alpha = 1, no_std = False):
        # assert TEST in self.data

        model_name = []
        tr_mean = []
        t_mean = []
        tr_variance = []
        t_variance = []

        # for model in self.data[TEST]:
        #     if models is not None and model not in models:
        #         continue
        #     test_data = self.data[TEST][model]
        #     model_name.append(model)
        #     tr_mean.append(np.mean(test_data[TestCharts.TRAIN_ACC]))
        #     t_mean.append(np.mean(test_data[TestCharts.VAL_ACC]))
        #     tr_variance.append(np.std(test_data[TestCharts.TRAIN_ACC]))    
        #     t_variance.append(np.std(test_data[TestCharts.VAL_ACC]))
        
        for model, model_data in self.results.test_data.items():
            if models is not None and model not in models:
                continue
           
            model_name.append(model)
            train_mean, train_std = model_data.get_train_mean_std()
            val_mean, val_std = model_data.get_val_mean_std()
            tr_mean.append(train_mean)
            t_mean.append(val_mean)
            tr_variance.append(train_std)    
            t_variance.append(val_std)

        fig, ax = self.get_plot(models, width_alpha=width_alpha)

        if no_std:
            ax.plot(model_name, tr_mean, 'ro', label="Train Accuracy")
            ax.plot(model_name, t_mean, 'go', label="Val Accuracy")
        else:
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

        # Combine training data
        new_results = self.results.shallow_merge_with(charts.results)

        new_chart = ChartUtil(results = new_results,
        data = combined_data
        )

        return new_chart
    
    def combine_chart_list(self, charts: list[ChartUtil], suffix = ""):
        results = [chart.results for chart in charts]
        new_results = self.results.shallow_merge_with_list(results)

        new_chart = ChartUtil(results = new_results)

        return new_chart

