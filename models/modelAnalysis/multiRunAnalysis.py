
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from numpy import ndarray
import torch

from metrics.chartSparseAnalysis import ChartSparseUtil
from models.modelAnalysis.modelAnalysis import ModelAnalysis
from models.modelAnalysis.similarityResult import SimilarityResult


@dataclass
class MultiRunAnalysis:
    """Main container for multi-run analysis across all models and layers"""
    models: Dict[str, ModelAnalysis] = field(default_factory=dict)
    chart_util: ChartSparseUtil = field(default=ChartSparseUtil()) 
    
    def __post_init__(self):
        if self.chart_util is None:
            self.chart_util = ChartSparseUtil()
    
    def get_or_create_model(self, model_name: str) -> ModelAnalysis:
        if model_name not in self.models:
            self.models[model_name] = ModelAnalysis(model_name)
        return self.models[model_name]
    
    def add_run_test_data(self, 
                          test_activations: Dict[str, Dict[str, torch.Tensor]],
                          compute_witihin_btw_sim_avg: Callable[..., tuple[float, float, float]],
                          compute_witihin_btw_sim_abs: Callable[..., tuple[dict[str, float], dict[str, float]]],
                          **kwargs):
        """Add data from a single run to the analysis"""
        for model_name, layer_data in test_activations.items():
            model_analysis = self.get_or_create_model(model_name)
            
            for layer_name, activations in layer_data.items():
                # Compute similarities
                num_inputs = activations.shape[0]
                cosine_sim_matrix = self.chart_util.get_cosine_similarity_matrix(activations, num_inputs)
                within_sim, between_sim, complete_sim = compute_witihin_btw_sim_avg(cosine_sim_matrix, num_inputs, **kwargs)
                within_sim_abs, between_sim_abs = compute_witihin_btw_sim_abs(cosine_sim_matrix, **kwargs)
                
                separation_value = 1 - between_sim
                
                similarity_result = SimilarityResult(
                    complete_similarity=complete_sim,
                    within_similarity_absolute=within_sim_abs,
                    between_similarity_absolute=between_sim_abs
                )
                
                model_analysis.add_test_run(
                    layer_name=layer_name,
                    integration=within_sim,
                    separation=separation_value,
                    similarity_result=similarity_result,
                    activations=activations
                )

    def add_run_test_data_xor(self, test_activations: Dict[str, Dict[str, torch.Tensor]]):
        self.add_run_test_data(
            test_activations,
            compute_witihin_btw_sim_avg=self.chart_util.compute_within_between_similarity_avg_for_xor,
            compute_witihin_btw_sim_abs=self.chart_util.compute_within_between_similarity_absolute_for_xor
        )

    def add_run_test_data_dummy(self, test_activations: Dict[str, Dict[str, torch.Tensor]], inputs: torch.Tensor, outputs: torch.Tensor):
        self.add_run_test_data(
            test_activations,
            compute_witihin_btw_sim_avg = self.chart_util.compute_within_between_similarity_avg_for_dummy,
            compute_witihin_btw_sim_abs = self.chart_util.compute_within_between_similarity_absolute_for_dummy,
            inputs = inputs,
            labels = outputs
        )

    def add_run_train_data(self, 
                           train_activations: Dict[str, Dict[int, Dict[str, torch.Tensor]]],
                           compute_witihin_btw_sim_avg: Callable[..., tuple[float, float, float]],
                           compute_witihin_btw_sim_abs: Callable[..., tuple[dict[str, float], dict[str, float]]],
                           **kwargs) :
        """{ model_name: { epoch_number: { layer_name: activations } } }"""
        for model_name, epoch_data in train_activations.items():
            model_analysis = self.get_or_create_model(model_name)
            for epoch_number, layer_data in epoch_data.items():
                for layer_name, activations in layer_data.items():
                    # Compute similarities
                    num_inputs = activations.shape[0]
                    cosine_sim_matrix = self.chart_util.get_cosine_similarity_matrix(activations, num_inputs)
                    within_sim, between_sim, complete_sim = compute_witihin_btw_sim_avg(cosine_sim_matrix, num_inputs, **kwargs)
                    within_sim_abs, between_sim_abs = compute_witihin_btw_sim_abs(cosine_sim_matrix, **kwargs)
                    
                    separation_value = 1 - between_sim
                    
                    similarity_result = SimilarityResult(
                        complete_similarity=complete_sim,
                        within_similarity_absolute=within_sim_abs,
                        between_similarity_absolute=between_sim_abs
                    )
                    
                    model_analysis.add_train_run(
                        epoch_number=epoch_number,
                        layer_name=layer_name,
                        integration=within_sim,
                        separation=separation_value,
                        similarity_result=similarity_result,
                        activations=activations
                    )
    
    def add_run_train_data_xor(self, train_activations: Dict[str, Dict[int, Dict[str, torch.Tensor]]]):
        """{ model_name: { epoch_number: { layer_name: activations } } }"""
        self.add_run_train_data(
            train_activations,
            compute_witihin_btw_sim_avg=self.chart_util.compute_within_between_similarity_avg_for_xor,
            compute_witihin_btw_sim_abs=self.chart_util.compute_within_between_similarity_absolute_for_xor
        )
    def add_run_train_data_dummy(self, train_activations: Dict[str, Dict[int, Dict[str, torch.Tensor]]], inputs: torch.Tensor, outputs: torch.Tensor):
        """{ model_name: { epoch_number: { layer_name: activations } } }"""
        self.add_run_train_data(
            train_activations,
            compute_witihin_btw_sim_avg = self.chart_util.compute_within_between_similarity_avg_for_dummy,
            compute_witihin_btw_sim_abs = self.chart_util.compute_within_between_similarity_absolute_for_dummy,
            inputs = inputs,
            labels = outputs
        )
    
    def generate_histogram_data(self, needIntegration: bool, 
                              model_names: Optional[List[str]] = None,
                              layer_names: Optional[List[str]] = None) -> Dict[str, Dict[str, List[float]]]:
        """Generate histogram data for plotting"""
        histogram_data: Dict[str, Dict[str, List[float]]] = {}
        
        target_models = model_names if model_names else list(self.models.keys())
        
        for model_name in target_models:
            if model_name not in self.models:
                continue
                
            model_analysis = self.models[model_name]
            histogram_data[model_name] = {}
            
            target_layers = layer_names if layer_names else list(model_analysis.test_analysis.keys())
            
            for layer_name in target_layers:
                if layer_name not in model_analysis.test_analysis:
                    continue
                    
                layer_analysis = model_analysis.test_analysis[layer_name]
                values = layer_analysis.get_integration_values() if needIntegration  else layer_analysis.get_separation_values()
                histogram_data[model_name][layer_name] = values
        
        return histogram_data