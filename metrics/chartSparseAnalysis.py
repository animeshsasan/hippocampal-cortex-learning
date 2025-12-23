from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.functional import F


class ChartSparseUtil():
    def __init__(self):
        self.allTestActivations = []
        self.allTrainActivations = []

    def add_test_activations(self, activation):
        self.allTestActivations.append(activation)
        self.layers = list(self.allTestActivations[0]['Complete Sparse model'].keys())
    
    def add_train_activations(self, activation):
        self.allTrainActivations.append(activation)
        self.layers = list(self.allTrainActivations[0]['Complete Sparse model'].keys())
        
    def _get_sparsity_across_layers(self, activations, zero_thresh = 1e-2, comparison_model = 'Dense Model'):
        sparsity_data = {model: {layer: [] for layer in self.layers} for model in self.allTestActivations[0].keys()}

        for sparseActivations in activations:
            for modelName in sparseActivations:
                for layerName in self.layers:
                    acts = sparseActivations[modelName][layerName]  # shape (num_inputs, num_units)
                    # Compute sparsity (fraction of near-zero activations)
                    sparsity = (acts.abs() < zero_thresh).float().mean().item()
                    sparsity_data[modelName][layerName].append(sparsity)

        # Average across runs
        avg_sparsity = {
            model: {layer: np.mean(sparsity_data[model][layer]) for layer in self.layers}
            for model in sparsity_data
        }

        # Compare to dense baseline
        dense_sparsity = avg_sparsity[comparison_model]
        sparsity_diff = {
            model: {layer: avg_sparsity[model][layer] - dense_sparsity[layer] for layer in self.layers}
            for model in avg_sparsity if model != comparison_model
        }
        return sparsity_data, avg_sparsity, sparsity_diff
     
    def set_test_sparsity_across_layers(self, zero_thresh = 1e-2, comparison_model = 'Dense Model'):
        self.test_sparsity_data, \
            self.test_avg_sparsity, \
                self.test_sparsity_diff = self._get_sparsity_across_layers(self.allTestActivations, zero_thresh, comparison_model)
    
    def set_train_sparsity_across_layers(self, zero_thresh = 1e-2, comparison_model = 'Dense Model'):
        self.train_sparsity_data, \
            self.train_avg_sparsity, \
                self.train_sparsity_diff \
                      = self._get_sparsity_across_layers(self.allTrainActivations["activations"], zero_thresh, comparison_model)

    def get_different_sparsity_graph_filters(self):
        controlModel = [model for model in self.test_avg_sparsity.keys() if 'Control' in model]
        denseModel = [model for model in self.test_avg_sparsity.keys() if 'Dense' in model]
        sparseModel = [model for model in self.test_avg_sparsity.keys() if 'Sparse' in model]

        return controlModel, denseModel, sparseModel
    
    def _plot_sparsity_across_layers(self, avg_sparsity, model_filter):
        fig, ax = plt.subplots(figsize=(15, 5))
        width = 0.25
        for _, model in enumerate(avg_sparsity.keys()):
            if 'Ortho' in model:
                continue
            if model not in model_filter:
                continue
            layers = list(avg_sparsity[model].keys())
            sparsity_vals = [avg_sparsity[model][layer] for layer in layers]
            xLabels = [model + " " + layer for layer in layers]
            ax.bar(xLabels, sparsity_vals, width)

        ax.set_title('Average sparsity')
        ax.legend()
        return fig
    
    def plot_test_sparsity_across_layers(self, model_filter):
        self._plot_sparsity_across_layers(self.test_avg_sparsity, model_filter)

    def plot_train_sparsity_across_layers(self, model_filter):
        self._plot_sparsity_across_layers(self.train_avg_sparsity, model_filter)

    def set_integration_separation(self):
        # Example usage after you compute cosine_sim_matrix:
        self.integration = {}
        self.separation = {}

        self.simmilarity_within = {}
        self.simmilarity_btw = {}
        for sparseActivations in self.allTestActivations:
            for modelName in sparseActivations:
                for layerName in self.layers:
                    acts = sparseActivations[modelName][layerName]  # shape: (4, 250)

                    # Compute cosine similarities
                    # We'll compare each pair of the 4 activations
                    num_inputs = acts.shape[0]
                    if modelName not in self.integration.keys():
                        self.integration[modelName] = {}
                        self.simmilarity_within[modelName] = {}
                    if layerName not in self.integration[modelName]:
                        self.integration[modelName][layerName] = []
                        self.simmilarity_within[modelName][layerName] = []
                    if modelName not in self.separation.keys():
                        self.separation[modelName] = {}
                        self.simmilarity_btw[modelName] = {}
                    if layerName not in self.separation[modelName]:
                        self.separation[modelName][layerName] = []
                        self.simmilarity_btw[modelName][layerName] = []

                    cosine_sim_matrix = self.get_cosine_similarity_matrix(acts, num_inputs)
                    within_sim, between_sim = self.compute_within_between_similarity_avg_for_xor(cosine_sim_matrix)
                    within_sim_abs, between_sim_abs = self.compute_within_between_similarity_absolute_for_xor(cosine_sim_matrix)

                    separation = 1 - between_sim
                    
                    self.integration[modelName][layerName].append(within_sim)
                    self.separation[modelName][layerName].append(separation)

                    self.simmilarity_within[modelName][layerName].append(within_sim_abs)
                    self.simmilarity_btw[modelName][layerName].append(between_sim_abs)
        self.combined_simmilarity = {}

        for key in self.simmilarity_within.keys():
            self.combined_simmilarity[key] = {}
            for subkey in self.simmilarity_within[key].keys():
                self.combined_simmilarity[key][subkey] = [
            {**i, **s} for i, s in zip(self.simmilarity_within[key][subkey], self.simmilarity_btw[key][subkey])
                ]
    def compute_within_between_similarity_avg(self, sim_matrix: np.ndarray, zero_categories: np.ndarray, one_categories: np.ndarray):
        within = []
        for i in range(len(zero_categories)):
            for j in range(i + 1, len(zero_categories)):
                within.append(sim_matrix[zero_categories[i], zero_categories[j]])
        
        for i in range(len(one_categories)):
            for j in range(i + 1, len(one_categories)):
                within.append(sim_matrix[one_categories[i], one_categories[j]])

        between = []
        for i in range(len(zero_categories)):
            for j in range(len(one_categories)):
                between.append(sim_matrix[zero_categories[i], one_categories[j]])

        complete = np.concat([within, between])
        # within = [sim_matrix[0, 3], sim_matrix[1, 2]]
        # between = [sim_matrix[0, 1], sim_matrix[0, 2], sim_matrix[1, 3], sim_matrix[2, 3]]
        return np.mean(within), np.mean(between), np.mean(complete)
           
    def compute_within_between_similarity_avg_for_xor(self, sim_matrix, num_inputs):
            even_categories = np.arange(num_inputs/2, dtype=np.int32) * 2
            odd_categories = np.arange(num_inputs/2, dtype=np.int32) * 2 + 1
            if num_inputs % 2 == 1:
                odd_categories = odd_categories[:-1]
            
            return self.compute_within_between_similarity_avg(sim_matrix, even_categories, odd_categories)
    
    def compute_within_between_similarity_avg_for_dummy(self, sim_matrix: np.ndarray, num_inputs: int, inputs:torch.Tensor, labels: torch.Tensor):
            zero_categories = torch.where(labels == 0)[0].numpy()
            one_categories = torch.where(labels == 1)[0].numpy()
            
            return self.compute_within_between_similarity_avg(sim_matrix, zero_categories, one_categories)
    
    def compute_within_between_similarity_absolute_for_xor(self, sim_matrix):
        within = {"00-11":sim_matrix[0, 3], "01-10": sim_matrix[1, 2]}
        between = {"00-01": sim_matrix[0, 1], "00-10": sim_matrix[0, 2], "11-01": sim_matrix[1, 3],"11-10": sim_matrix[2, 3]}
        self.integration_labels_abs = ["00-11", "01-10"]
        self.separation_labels_abs = ["00-01", "00-10", "11-01", "11-10"]
        return within, between
    
    def compute_within_between_similarity_absolute_for_dummy(self, sim_matrix: np.ndarray, inputs: torch.Tensor, labels: torch.Tensor):
        zero_categories = torch.where(labels == 0)[0].numpy()
        one_categories = torch.where(labels == 1)[0].numpy()
        self.integration_labels_abs =[]
        self.separation_labels_abs = []
        within = {}
        for i in range(len(zero_categories)):
            for j in range(i + 1, len(zero_categories)):
                idx_1 = zero_categories[i]
                idx_2 = zero_categories[j]
                within[str(inputs[idx_1]) + "-" + str(inputs[idx_2])] = sim_matrix[idx_1, idx_2]
                self.integration_labels_abs.append(str(inputs[idx_1]) + "-" + str(inputs[idx_2]))
        for i in range(len(one_categories)):
            for j in range(i + 1, len(one_categories)):
                idx_1 = one_categories[i]
                idx_2 = one_categories[j]
                within[str(inputs[idx_1]) + "-" + str(inputs[idx_2])] = sim_matrix[idx_1, idx_2]
                self.integration_labels_abs.append(str(inputs[idx_1]) + "-" + str(inputs[idx_2]))

        between = {}
        for i in range(len(zero_categories)):
            for j in range(len(one_categories)):
                idx_1 = zero_categories[i]
                idx_2 = one_categories[j]
                between[str(inputs[idx_1]) + "-" + str(inputs[idx_2])] = sim_matrix[idx_1, idx_2]
                self.separation_labels_abs.append(str(inputs[idx_1]) + "-" + str(inputs[idx_2]))

        return within, between

    def get_cosine_similarity_matrix(self, acts, num_inputs = 4) -> np.ndarray:
        cosine_sim_matrix = torch.zeros((num_inputs, num_inputs))
        for i in range(num_inputs):
            for j in range(num_inputs):
                cosine_sim_matrix[i, j] = F.cosine_similarity(acts[i], acts[j], dim=0)
        return cosine_sim_matrix.numpy()

    def get_similarity_plot(self, acts, modelName, layerName, input_labels, num_inputs=4, fig=None, ax=None):
        # Compute cosine similarity matrix
        cosine_sim_matrix = self.get_cosine_similarity_matrix(acts, num_inputs)

        # Create figure and axes if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot heatmap
        im = ax.imshow(cosine_sim_matrix, cmap='viridis', interpolation='nearest')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity')

        # Set ticks and labels
        ax.set_xticks(range(num_inputs))
        ax.set_yticks(range(num_inputs))
        ax.set_xticklabels(input_labels, rotation=45, ha='right')
        ax.set_yticklabels(input_labels)
        ax.set_title(f'Cosine Similarity Heatmap for {modelName} and Layer "{layerName}"')
        ax.set_aspect('equal')

        return fig
    
    def get_line_chart(self, yData, graphLabel):
        figures = {}
        xData = np.arange(len(self.allTestActivations))

        for modelName in yData:
            fig, ax = plt.subplots()

            figures[modelName] = fig

            ax.set_xlabel("Run Number")
            ax.set_ylabel(graphLabel)

            for layerName in yData[modelName]:
                ax.plot(xData, yData[modelName][layerName], label=layerName)

            ax.legend()
            ax.set_title(graphLabel + " " + modelName)
            

        return figures
    
    def get_integration_line_chart_across_runs(self):
        return self.get_line_chart(self.integration, "Integration")
    
    def get_separation_line_chart_across_runs(self):
        return self.get_line_chart(self.separation, "Separation")
    
    def get_simmilarity_line_chart(self, xData, yDate, titlePrefix):
        figures = {}
        xData = np.arange(len(self.allTestActivations))

        for modelName in yDate:
            for layerName in yDate[modelName]:
                fig, ax = plt.subplots()
                if modelName not in figures.keys():
                    figures[modelName] = {}

                figures[modelName][layerName] = fig
                ax.set_xlabel("Run Number")
                ax.set_ylabel("Cosine Simmilarity")

                values = {}
                for abs_values in yDate[modelName][layerName]:
                    for value_key in abs_values:
                        if value_key not in values.keys():
                                values[value_key] = []
                        values[value_key].append(abs_values[value_key])
                for value_key in values:
                    ax.plot(xData, values[value_key], label=value_key)

                ax.legend()
                ax.set_title(titlePrefix + " " + modelName + " " + layerName)
            

        return figures
    
    def get_simmilarity_btw_line_chart_abs(self):
        return self.get_simmilarity_line_chart(self.simmilarity_btw, "Separation")
    
    def get_simmilarity_within_line_chart_abs(self):
        return self.get_simmilarity_line_chart(self.simmilarity_within, "Integrations")

    def get_error_bar_chart(yDataList, titlePrefix):
        figures = {}
        for modelName in yDataList:
            fig, ax = plt.subplots()

            if modelName not in figures.keys():
                figures[modelName] = {}

            figures[modelName] = fig

            ax.set_xlabel("Layer Name")
            mean_vals = []
            std_vals = []

            for layerName in yDataList[modelName]:
                mean = np.mean(yDataList[modelName][layerName])
                std_dev = np.std(yDataList[modelName][layerName])
                mean_vals.append(mean)
                std_vals.append(std_dev)

            ax.errorbar(list(yDataList[modelName].keys())  , y = mean_vals, yerr= std_vals)
            ax.legend()
            ax.set_title(titlePrefix + " " + modelName)

        return figures
    
    def get_integration_error_bar_chart(self):
        return self.get_error_bar_chart(self.integration, "Integration")
    
    def get_separation_error_bar_chart(self):
        return self.get_error_bar_chart(self.separation, "Separation")
    
    def get_error_bar_chart_input_pair(self):
            figures = {}
            for modelName in self.combined_simmilarity:
                fig, ax = plt.subplots()
                if modelName not in figures.keys():
                    figures[modelName] = {}
                figures[modelName] = fig
                ax.set_xlabel("Input Pair")
                ax.set_ylabel("Cosine Simmilarity")

                for layerName in self.combined_simmilarity[modelName]:
                    mean_vals = []
                    std_vals = []
                    labels = []
                    layer_dicts = self.combined_simmilarity[modelName][layerName]
                    for label in layer_dicts[0].keys():
                        items = [item[label] for item in layer_dicts]

                        mean = np.mean(items)
                        std_dev = np.std(items)

                        mean_vals.append(mean)
                        std_vals.append(std_dev)

                    labels = list(layer_dicts[0].keys())
                    ax.errorbar(labels , y = mean_vals, yerr= std_vals, label = layerName)
                ax.legend()
                ax.set_title(modelName)

            return figures
                        