from enum import Enum
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats # type: ignore[import]
from metrics.bootstrapUtils import get_bootstrapped_value
from models.enums.similarityType import AbsSimilarityType
from models.modelAnalysis.layerAnalysis import LayerAnalysis
from models.modelAnalysis.multiRunAnalysis import MultiRunAnalysis

class AvgSimilarityValues(Enum):
        INTEGRATION = "integration"
        SEPARATION = "separation"
        COMPLETE = "complete"

class ModelAnalysisUtils():
    def create_test_similarity_complete_df_multi_run(self, analysis: MultiRunAnalysis) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create DataFrame in the same format as your original code but with means across runs
        """
        similarity_data: dict[str, dict[str, float]] = {}
        ci_data: dict[str, dict[str, float]] = {}

        for model_name, model_analysis in analysis.models.items():
            similarity_data[model_name] = {}
            ci_data[model_name] = {}
            for layer_name, layer_analysis in model_analysis.test_analysis.items():
                # Get mean complete similarity across all runs
                complete_similarities = layer_analysis.get_complete_similarity_values()
                mean_similarity, ci = self._calculate_mean_and_ci(complete_similarities)
                
                similarity_data[model_name][layer_name] = mean_similarity
                ci_data[model_name][layer_name] = ci
        
        # Convert to DataFrame (this matches your original format)
        df_mean = pd.DataFrame(similarity_data).T
        df_ci = pd.DataFrame(ci_data).T

        return df_mean, df_ci
    
    def create_train_similarity_complete_df_multi_run(self, analysis: MultiRunAnalysis) -> Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
        # Training data (line plots) - structure: {model_name: {layer_name: {epoch: mean_value}}}
        return self._create_train_similarity_df_multi_run(analysis, AvgSimilarityValues.COMPLETE)
    
    def create_train_int_sep_df_multi_run(self, analysis: MultiRunAnalysis, needIntegration: bool = True) -> Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
        valueNeeded = AvgSimilarityValues.INTEGRATION if needIntegration else AvgSimilarityValues.SEPARATION

        return self._create_train_similarity_df_multi_run(analysis, valueNeeded)
    
    def _create_train_similarity_df_multi_run(self, analysis: MultiRunAnalysis, valueType: AvgSimilarityValues) -> Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
        # Training data (line plots) - structure: {model_name: {layer_name: {epoch: mean_value}}}
        train_similarity_data: dict[str, dict[str, dict[int, float]]] = {}
        train_ci_data: dict[str, dict[str, dict[int, float]]] = {}

        for model_name, model_analysis in analysis.models.items():
            train_similarity_data[model_name] = {}
            train_ci_data[model_name] = {}
        
            epoch_numbers = model_analysis.get_epoch_numbers()
            for epoch in epoch_numbers:
                epoch_analysis = model_analysis.train_analysis[epoch]
                for layer_name, layer_analysis in epoch_analysis.layers.items():
                    if layer_name not in train_similarity_data[model_name]:
                        train_similarity_data[model_name][layer_name] = {}
                        train_ci_data[model_name][layer_name] = {}
                    
                    complete_similarities = self._get_values_from_layer(layer_analysis, valueType)
                    mean_similarity, ci = self._calculate_mean_and_ci(complete_similarities)
                    
                    train_similarity_data[model_name][layer_name][epoch] = mean_similarity
                    train_ci_data[model_name][layer_name][epoch] = ci


        return self._convert_train_data_to_dataframes(train_similarity_data, train_ci_data)
    
    def _get_values_from_layer(self, layer_analysis: LayerAnalysis, valueType: AvgSimilarityValues) -> List[float]:
        if valueType == AvgSimilarityValues.INTEGRATION:
            return layer_analysis.integration_values
        elif valueType == AvgSimilarityValues.SEPARATION:
            return layer_analysis.separation_values
        elif valueType == AvgSimilarityValues.COMPLETE:
            return layer_analysis.get_complete_similarity_values()
    
    def _convert_train_data_to_dataframes(self, train_similarity_data: dict, train_ci_data: dict) -> Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
        Convert training data to a nested dictionary of DataFrames
        Returns: {model_name: {layer_name: (df_mean, df_ci)}}
        """
        train_dfs: Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]] = {}
        
        for model_name, layer_data in train_similarity_data.items():
            train_dfs[model_name] = {}
            
            for layer_name, epoch_data in layer_data.items():
                epochs = sorted(epoch_data.keys())
                
                # Extract mean values and CI values in epoch order
                mean_values = [epoch_data[epoch] for epoch in epochs]
                ci_values = [train_ci_data[model_name][layer_name][epoch] for epoch in epochs]
                
                # Create DataFrames
                df_mean = pd.DataFrame({
                    'epoch': epochs,
                    'similarity': mean_values
                }).set_index('epoch')
                
                df_ci = pd.DataFrame({
                    'epoch': epochs,
                    'ci': ci_values
                }).set_index('epoch')
                
                train_dfs[model_name][layer_name] = (df_mean, df_ci)
        
        return train_dfs

    def _calculate_mean_and_ci(self, values: List[float]) -> Tuple[float, float]:
        """Calculate mean and 95% CI for a list of values"""
        if len(values) == 0:
            return 0.0, 0.0
        elif len(values) == 1:
            return float(values[0]), 0.0
        else:
            mean_similarity = float(np.mean(values))
            sem = stats.sem(values, nan_policy='omit')
            ci = 1.96 * sem
            return mean_similarity, ci
        
    def _calculate_mean_and_bt(self, values: List[float]) -> Tuple[float, float, float]:
        """Calculate mean and 95% CI for a list of values"""
        if len(values) == 0:
            return 0.0, 0.0, 0.0
        elif len(values) == 1:
            return float(values[0]), 0.0, 0.0
        else:
            mean_similarity = float(np.mean(values))
            low, high = get_bootstrapped_value(values)
            return mean_similarity, low, high

    def create_test_int_sep_df_multi_run(self, analysis: MultiRunAnalysis, needWithin: bool = True, needBootstrapped = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        similarity_data: dict[str, dict[str, float]] = {}
        ci_data: dict[str, dict[str, float]] = {}

        for model_name, model_analysis in analysis.models.items():
            similarity_data[model_name] = {}
            ci_data[model_name] = {}
            for layer_name, layer_analysis in model_analysis.test_analysis.items():
                # Get mean complete similarity across all runs
                within_btw_values = layer_analysis.integration_values if needWithin else [ 1 - val for val in layer_analysis.separation_values]
                # if needBootstrapped:
                #     # mean_similarity, l_b, u_b = self._calculate_mean_and_bt(within_btw_values, needBootstrapped=True)
                # else:
                mean_similarity, ci = self._calculate_mean_and_ci(within_btw_values)

                similarity_data[model_name][layer_name] = mean_similarity
                ci_data[model_name][layer_name] = ci
        
        # Convert to DataFrame (this matches your original format)
        df_mean = pd.DataFrame(similarity_data).T
        df_ci = pd.DataFrame(ci_data).T

        df_mean = df_mean.reset_index().rename(columns={"index": "Model"})
        df_ci = df_ci.reset_index().rename(columns={"index": "Model"})

        return df_mean, df_ci

    def create_test_absolute_sim_graph(self, analysis: MultiRunAnalysis, graphType: AbsSimilarityType = AbsSimilarityType.BETWEEN) -> pd.DataFrame:
        records = []

        for model_name, model_analysis in analysis.models.items():
            for layer_name, layer_analysis in model_analysis.test_analysis.items():
                # Get mean complete similarity across all runs
                abs_similarity = layer_analysis.get_absolute_similarity_values(graphType)
                for runAnanlysis in abs_similarity:
                    for cond, value in runAnanlysis.items():
                        records.append({
                        "Model": model_name,
                        "Layer": layer_name,
                        "Condition": cond,
                        "Value": value
                    })


        df = pd.DataFrame(records)

        return df
    
    def create_train_absolute_sim_graph(self, analysis: MultiRunAnalysis, graphType: AbsSimilarityType = AbsSimilarityType.BETWEEN) -> pd.DataFrame:
        records = []

        for model_name, model_analysis in analysis.models.items():
            epoch_numbers = model_analysis.get_epoch_numbers()
            for epoch in epoch_numbers:
                epoch_analysis = model_analysis.train_analysis[epoch]
                for layer_name, layer_analysis in epoch_analysis.layers.items():
                    abs_similarity = layer_analysis.get_absolute_similarity_values(graphType)
                    for runAnanlysis in abs_similarity:
                        for cond, value in runAnanlysis.items():
                            records.append({
                            "Model": model_name,
                            "Layer": layer_name,
                            "Condition": cond,
                            "Value": value,
                            "Epoch": epoch
                        })

        df = pd.DataFrame(records)

        return df
    
    def create_train_avg_sim_graph(self, analysis: MultiRunAnalysis, needWithin = True) -> pd.DataFrame:
        records = []

        for model_name, model_analysis in analysis.models.items():
            epoch_numbers = model_analysis.get_epoch_numbers()
            for epoch in epoch_numbers:
                epoch_analysis = model_analysis.train_analysis[epoch]
                for layer_name, layer_analysis in epoch_analysis.layers.items():
                    abs_similarity = layer_analysis.get_integration_values() if needWithin else [ 1 - val for val in layer_analysis.get_separation_values()]
                    for value in abs_similarity:
                        records.append({
                            "Model": model_name,
                            "Layer": layer_name,
                            "Value": value,
                            "Epoch": epoch,
                            "Condition": "Within" if needWithin else "Between"
                        })

        df = pd.DataFrame(records)

        return df