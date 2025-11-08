from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import torch
from models.enums.similarityType import AbsSimilarityType
from models.modelAnalysis.similarityResult import SimilarityResult
import numpy as np


@dataclass
class LayerAnalysis:
    """Analysis results for a single layer across multiple runs"""
    layer_name: str
    integration_values: List[float] = field(default_factory=list)
    separation_values: List[float] = field(default_factory=list)
    similarity_results: List[SimilarityResult] = field(default_factory=list)
    activation_values: List[torch.Tensor] = field(default_factory=list)
    
    def add_run(self, 
                integration: float, 
                separation: float, 
                similarity_result: SimilarityResult, 
                activation: torch.Tensor):
        self.integration_values.append(integration)
        self.separation_values.append(separation)
        self.similarity_results.append(similarity_result)
        self.activation_values.append(activation)
    
    def get_integration_mean_std(self) -> Tuple[np.float32, np.float32]:
        return np.mean(self.integration_values), np.std(self.integration_values)
    
    def get_separation_mean_std(self) -> Tuple[np.float32, np.float32]:
        return np.mean(self.separation_values), np.std(self.separation_values)
    
    def get_absolute_similarity_values(self, similarity_type: AbsSimilarityType) -> List[Dict[str, float]]:
        if similarity_type == AbsSimilarityType.WITHIN:
            return [result.within_similarity_absolute for result in self.similarity_results ]
        elif similarity_type == AbsSimilarityType.BETWEEN:
            return [result.between_similarity_absolute for result in self.similarity_results]
    
    def get_complete_similarity_values(self) -> List[float]:
        return [result.complete_similarity for result in self.similarity_results]
    
    
    def get_integration_values(self) -> List[float]:
        return self.integration_values
    
    def get_separation_values(self) -> List[float]:
        return self.separation_values