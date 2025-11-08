from dataclasses import dataclass, field
from typing import Dict

from models.trainingAnalysis.modelTestData import ModelTestData
from models.trainingAnalysis.modelTrainingData import ModelTrainingData



@dataclass
class ExperimentResults:
    """Main container for all training and test results"""
    train_data: Dict[str, ModelTrainingData] = field(default_factory=dict)
    test_data: Dict[str, ModelTestData] = field(default_factory=dict)
    
    def get_or_create_training_model(self, model: str) -> ModelTrainingData:
        """Get existing training data for model or create new one"""
        if model not in self.train_data:
            self.train_data[model] = ModelTrainingData()
        return self.train_data[model]
    
    def get_or_create_test_model(self, model: str) -> ModelTestData:
        """Get existing test data for model or create new one"""
        if model not in self.test_data:
            self.test_data[model] = ModelTestData()
        return self.test_data[model]