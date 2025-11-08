from dataclasses import dataclass, field
from typing import Dict

import torch

from models.modelAnalysis.layerAnalysis import LayerAnalysis
from models.modelAnalysis.similarityResult import SimilarityResult


@dataclass
class EpochAnalysis:
    """Analysis results for a specific epoch"""
    epoch_number: int
    layers: Dict[str, LayerAnalysis] = field(default_factory=dict)
    
    def get_or_create_layer(self, layer_name: str) -> LayerAnalysis:
        if layer_name not in self.layers:
            self.layers[layer_name] = LayerAnalysis(layer_name)
        return self.layers[layer_name]
    
    def add_layer_run(self, 
                      layer_name: str, 
                      integration: float, 
                      separation: float, 
                      similarity_result: SimilarityResult,
                      activations: torch.Tensor):
        layer_analysis = self.get_or_create_layer(layer_name)
        layer_analysis.add_run(integration, separation, similarity_result, activations)