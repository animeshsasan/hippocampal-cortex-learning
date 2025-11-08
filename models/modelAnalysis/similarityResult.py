from typing import Dict
from dataclasses import dataclass


@dataclass
class SimilarityResult:
    """Stores similarity results for a single run"""
    complete_similarity: float
    within_similarity_absolute: Dict[str, float]
    between_similarity_absolute: Dict[str, float]
    
    def to_dict(self):
        return {
            'complete': self.complete_similarity,
            'within_absolute': self.within_similarity_absolute,
            'between_absolute': self.between_similarity_absolute
        }