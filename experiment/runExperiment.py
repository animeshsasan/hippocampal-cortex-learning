from typing import Callable
from metrics.charts import ChartUtil
from training.train import TrainUtil
import torch.nn as nn


class RunExperiment():
    def __init__(self, trainUtil: TrainUtil):
        self.trainUtil = trainUtil

    def perform_one_experiment(self,
                               models: dict[str, Callable], 
                               n_epochs = 41, 
                               n_runs=60,
                               ortho_lambda = 0.1,
                               in_features = 2, 
                               out_features = 2, 
                               layers = 3, 
                               model_params: dict[str, tuple[int, int | None]] = {"l1": (200, 15), "l2": (250, 20), "l3": (200, 10)},
                               print_summary = False,
                               batch_train = False) -> ChartUtil:
        chartUtil = ChartUtil()
        epoch_num = n_epochs
        for _ in range(n_runs):
            for modelName, get_model in models.items():
                model = get_model(in_features = in_features, out_features = out_features, layers = layers, model_params = model_params)

                if ortho_lambda and "Ortho" in modelName:
                    loss_over_epochs, accuracy_over_epochs, epoch_numbers, train_accuracy, val_accuracy = self.trainUtil.train_and_evaluate(model, 
                                                                                                                                epochs = epoch_num, 
                                                                                                                                name = modelName,
                                                                                                                                print_msg = print_summary,
                                                                                                                                batch_train = batch_train,
                                                                                                                                ortho_lambda = ortho_lambda)

                else:
                    loss_over_epochs, accuracy_over_epochs, epoch_numbers, train_accuracy, val_accuracy = self.trainUtil.train_and_evaluate(model, 
                                                                                                                                epochs = epoch_num, 
                                                                                                                                name = modelName,
                                                                                                                                print_msg = print_summary,
                                                                                                                                batch_train = batch_train)
                chartUtil.add_train_data(modelName, loss_over_epochs, accuracy_over_epochs, epoch_numbers)
                chartUtil.add_test_data(modelName, train_accuracy, val_accuracy)
        
        return chartUtil