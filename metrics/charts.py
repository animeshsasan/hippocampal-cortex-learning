from __future__ import annotations
import textwrap
from matplotlib import cm
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt

from models.enums.chartType import TrainingCharts
from models.trainingAnalysis.experimentResults import ExperimentResults

TEST = "test"
TRAIN = "train"


class ChartUtil():
    def __init__(self, results: ExperimentResults | None = None):
        self.colorsMap: dict[str, Colormap] = {}
        if results is None:
            self.results = ExperimentResults()
        else:
            self.results = results

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

    
    def add_train_data(self, model, loss_values, train_accuracy_over_epochs, epoch_numbers, test_accuracy_over_epochs):

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
                               set_unique_colors = False,
                               ax = None,
                               linestyle = "-"):
        # assert TRAIN in self.data
        if ax:
            fig = ax.figure
        else:
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
                    ax.plot(epochs, mean_acc, label=model, color = colorsMap[model], linestyle = linestyle)
                else:
                    ax.errorbar(epochs, mean_acc, std_acc, label=model, color = colorsMap[model], linestyle = linestyle)
            else:
                acc = []
                for epoch in model_data.epochs:
                    epoch_data = model_data.epochs[epoch]
                    acc_value = epoch_data.get_value(value_to_plot, run)
                    acc.append(acc_value)
                
                ax.plot(epochs, acc, label=model)
        
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
    
    def plot_multiple_training_data_for(self,
                                        ax,
                                        value_to_plot = TrainingCharts.LOSS, 
                                        models = None, 
                                        run = None, 
                                        width_alpha = 0.3, 
                                        height_alpha = 0.3, 
                                        bootstrap = False,
                                        set_unique_colors = False,
                                        linestyle = "-",
                                        color = None,
                                        epochs_to_plot = None,):
        if ax:
            fig = ax.figure
        else:
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
            if epochs_to_plot:
                epochs = epochs[:epochs_to_plot]
            plot_color = color if color != None else colorsMap[model]

            if run is None:
                mean_acc = []
                t_bt_l = []
                t_bt_h = []
                for epoch in epochs:
                    epoch_data = model_data.epochs[epoch]
                    
                    mean, bt_l, bt_h = epoch_data.get_value_mean_bootstrap_by_type(value_to_plot)
                    mean_acc.append(mean)
                    if bootstrap:
                        t_bt_l.append(bt_l)
                        t_bt_h.append(bt_h)

                ax.plot(epochs, mean_acc, label=model, color = plot_color, linestyle = linestyle)

                if bootstrap:
                    ax.fill_between(
                        epochs,
                        t_bt_l,
                        t_bt_h,
                        alpha=0.25,
                        color = plot_color
                    ) 
            else:
                acc = []
                for epoch in model_data.epochs:
                    epoch_data = model_data.epochs[epoch]
                    acc_value = epoch_data.get_value(value_to_plot, run)
                    acc.append(acc_value)
                
                ax.plot(epochs, acc, label=model)
        
        return fig
        
    def add_test_data(self, model, train_accuracy, val_accuracy):

        model_test_data = self.results.get_or_create_test_model(model)
        model_test_data.add_test_data(train_accuracy, val_accuracy)

    def plot_test_accu_for_models(self, models = None, width_alpha = 1, no_std = False):

        model_name = []
        tr_mean = []
        t_mean = []
        tr_variance = []
        t_variance = []
        
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
    
    def plot_test_accu_for_models_at_epoch(self, 
                                           epoch, 
                                           models = None, 
                                           width_alpha = 1, 
                                           needBootstrap = False,
                                           linestyle = '-',
                                           label = None,
                                           ax = None,
                                           color = None,
                                           x_axis_filter = None):

        model_name = []
        t_mean = []
        t_bt_l = []
        t_bt_h = []
        kwargs = {}
        if color:
            kwargs["color"] = color

        for model, model_data in self.results.train_data.items():
            if models is not None and model not in models:
                continue

            epoch_data = model_data.epochs[epoch]
            if x_axis_filter:
                model_name.append(x_axis_filter(model))
            else:
                model_name.append(model)
            val_mean, bt_l, bt_h = epoch_data.get_value_mean_bootstrap_by_type(TrainingCharts.VAL_ACC)

            t_mean.append(val_mean)
            if needBootstrap:
                t_bt_l.append(bt_l)
                t_bt_h.append(bt_h)

        if ax is None:
            fig, ax = self.get_plot(models, width_alpha=width_alpha)
        else:
            fig = ax.figure

        if label != None:
            combined_label = label + " Accuracy"

        ax.plot(model_name, t_mean, marker='o', linestyle=linestyle, label=combined_label, **kwargs)
        if needBootstrap:
            ax.fill_between(
                model_name,
                t_bt_l,
                t_bt_h,
                alpha=0.25,
                **kwargs
            )   
        ax.set_xlabel("Model Width")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_title(f"Accuracy at epoch {epoch}")

        return fig
    
    def get_wrapped_name(self, name, label_width):
        return "\n".join(textwrap.wrap(name, width=label_width))
    
    
    def plot_epochs_to_thr_for_models(self, 
                                           models = None,
                                           modelToColor = None, 
                                           width_alpha = 1, 
                                           needBootstrap = False,
                                           linestyle = '-',
                                           label = None,
                                           ax = None,
                                           lineColor = None,
                                           x_axis_filter = None,
                                           thrs = [95.],
                                           label_width = 12):

        model_name = []
        kwargs = {}
        means = {thr: [] for thr in thrs}
        bt_ls = {thr: [] for thr in thrs}
        bt_hs = {thr: [] for thr in thrs}
        if lineColor:
            kwargs["color"] = lineColor

        models_to_plot = models if models else self.results.train_data.keys()
        for model in models_to_plot:
            model_data = self.results.train_data[model]
            if models is not None and model not in models:
                continue
            if x_axis_filter:
                model_name.append(x_axis_filter(model))
                if modelToColor:
                    modelToColor[x_axis_filter(model)] = modelToColor[model]
            else:
                model_name.append(model)

            mean, bt_l, bt_h = model_data.get_epochs_to_thr_mean_bt(model, thrs)

            for thr in thrs:
                means[thr].append(mean[thr])
                bt_ls[thr].append(bt_l[thr])
                bt_hs[thr].append(bt_h[thr])

        if ax is None:
            fig, ax = self.get_plot(models, width_alpha=width_alpha)
        else:
            fig = ax.figure

        for thr in thrs:
            combined_label = ""
            if label != None:
                combined_label = label
            wrapped_names = [self.get_wrapped_name(name, label_width) for name in model_name]
            ax.plot(wrapped_names, means[thr], linestyle=linestyle, label=combined_label, **kwargs)
            for i, name in enumerate(model_name):
                # ax.scatter(
                #     [name],
                #     [means[thr][i]],
                #     marker='o',
                #     color=modelToColor[name],   # spreads colors nicely
                #     zorder=3
                # )
                if needBootstrap:
                    err_lower = means[thr][i] - bt_ls[thr][i]
                    err_upper = bt_hs[thr][i] - means[thr][i]
                    wrapped_name = self.get_wrapped_name(name, label_width)
                    ax.errorbar(
                        wrapped_name,
                        means[thr][i],
                        yerr=[[err_lower], [err_upper]],
                        fmt='o',
                        markersize=6,     
                        markeredgewidth=1,
                        markeredgecolor=modelToColor[name],  # marker edge
                        markerfacecolor=modelToColor[name],  # marker fill (this was missing!)
                        ecolor=modelToColor[name],
                        elinewidth=1,
                        capsize=3,
                        alpha=0.8
                    ) 
                else:
                    ax.scatter(
                    [name],
                    [means[thr][i]],
                    marker='o',
                    color=modelToColor[name],   # spreads colors nicely
                    zorder=3
                )
            # if needBootstrap:
            #     ax.fill_between(
            #         model_name,
            #         bt_ls[thr],
            #         bt_hs[thr],
            #         alpha=0.25,
            #         **kwargs
            #     )

        return fig
    
    def combine_charts(self, charts: ChartUtil):
        new_results = self.results.shallow_merge_with(charts.results)

        new_chart = ChartUtil(results = new_results
        )

        return new_chart
    
    def combine_chart_list(self, charts: list[ChartUtil]):
        results = [chart.results for chart in charts]
        new_results = self.results.shallow_merge_with_list(results)

        new_chart = ChartUtil(results = new_results)

        return new_chart

