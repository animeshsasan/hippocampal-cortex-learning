from dataclasses import dataclass, field
from typing import Dict, List

import torch

from models.modelAnalysis.epochAnalysis import EpochAnalysis
from models.modelAnalysis.layerAnalysis import LayerAnalysis
from models.modelAnalysis.similarityResult import SimilarityResult


@dataclass
class ModelAnalysis:
    """Analysis results for a single model across multiple runs"""
    model_name: str
    test_analysis: Dict[str, LayerAnalysis] = field(default_factory=dict)
    train_analysis: Dict[int, EpochAnalysis] = field(default_factory=dict)
    
    def get_or_create_layer(self, layer_name: str) -> LayerAnalysis:
        if layer_name not in self.test_analysis:
            self.test_analysis[layer_name] = LayerAnalysis(layer_name)
        return self.test_analysis[layer_name]

    def get_or_create_epoch(self, epoch_number: int) -> EpochAnalysis:
        if epoch_number not in self.train_analysis:
            self.train_analysis[epoch_number] = EpochAnalysis(epoch_number)
        return self.train_analysis[epoch_number]
    
    def add_test_run(self, 
                      layer_name: str, 
                      integration: float, 
                      separation: float, 
                      similarity_result: SimilarityResult,
                      activations: torch.Tensor):
        layer_analysis = self.get_or_create_layer(layer_name)
        layer_analysis.add_run(integration, separation, similarity_result, activations)
    
    def add_train_run(self, 
                      epoch_number: int,
                      layer_name: str, 
                      integration: float, 
                      separation: float, 
                      similarity_result: SimilarityResult,
                      activations: torch.Tensor):
        epoch_analysis = self.get_or_create_epoch(epoch_number)
        epoch_analysis.add_layer_run(layer_name, integration, separation, similarity_result, activations)
    
    def get_epoch_numbers(self) -> List[int]:
        return sorted(self.train_analysis.keys())