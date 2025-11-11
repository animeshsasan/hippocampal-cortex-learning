from __future__ import annotations
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
    
    def shallow_merge_with(self, other: ExperimentResults) -> ExperimentResults:

        new_train_data = self.train_data | other.train_data
        new_test_data = self.test_data | other.test_data
        new_experimental_result = ExperimentResults(new_train_data, new_test_data)
        return new_experimental_result
    
    def shallow_merge_with_list(self, others: list[ExperimentResults]) -> ExperimentResults:
        assert len(others) != 0

        new_train_data = self.train_data | others[0].train_data
        new_test_data = self.test_data | others[0].test_data

        for i in range(1, len(others)):
            new_train_data = self.train_data | others[i].train_data
            new_test_data = self.test_data | others[i].test_data
        new_experimental_result = ExperimentResults(new_train_data, new_test_data)
        return new_experimental_result
    
    def get_models(self) -> list[str]:
        train_models = list(self.train_data.keys())
        test_models = list(self.test_data.keys())
        if len(train_models) > len(test_models):
            return train_models
        return test_models
