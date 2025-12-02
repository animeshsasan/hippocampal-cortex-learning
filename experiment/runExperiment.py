from typing import Callable
from metrics.charts import ChartUtil
from training.train import TrainUtil
from utils.seedGenerator import generate_seeds
import torch
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
                               batch_train = False,
                               random_sequencing = True,
                               seed = 0,
                               activation = nn.LeakyReLU(0.1),
                               lr = 0.01) -> ChartUtil:
        chartUtil = ChartUtil()
        epoch_num = n_epochs
        seeds = generate_seeds(seed, n_runs)

        for i in range(n_runs):

            torch.manual_seed(seeds[i])

            for modelName, get_model in models.items():
                model = get_model(
                    in_features = in_features, 
                    out_features = out_features,
                    layers = layers, 
                    model_params = model_params,
                    activation = activation)

                if ortho_lambda and "Ortho" in modelName:
                    loss_over_epochs, \
                        train_accuracy_over_epochs, \
                            epoch_numbers, \
                                train_accuracy, \
                                    val_accuracy, \
                                        test_accuracy_over_epochs, \
                                            _,_ \
                                            = self.trainUtil.train_and_evaluate(model, 
                                                                                epochs = epoch_num, 
                                                                                name = modelName,
                                                                                print_msg = print_summary,
                                                                                batch_train = batch_train,
                                                                                ortho_lambda = ortho_lambda,
                                                                                random_sequencing = random_sequencing,
                                                                                lr = lr)

                else:
                    loss_over_epochs, \
                        train_accuracy_over_epochs, \
                            epoch_numbers, \
                                train_accuracy, \
                                    val_accuracy, \
                                        test_accuracy_over_epochs, \
                                            _,_ \
                                                      = self.trainUtil.train_and_evaluate(model, 
                                                                                epochs = epoch_num, 
                                                                                name = modelName,
                                                                                print_msg = print_summary,
                                                                                batch_train = batch_train,
                                                                                random_sequencing = random_sequencing,
                                                                                lr = lr)
                chartUtil.add_train_data(modelName, loss_over_epochs, train_accuracy_over_epochs, epoch_numbers, test_accuracy_over_epochs)
                chartUtil.add_test_data(modelName, train_accuracy, val_accuracy)
        
        return chartUtil
    
    def train_models(self,
                     models: dict[str, Callable],
                     n_epochs = 41,
                     ortho_lambda = 0.1,
                     in_features = 2, 
                     out_features = 2, 
                     layers = 3, 
                     model_params: dict[str, tuple[int, int | None]] = {"l1": (200, 15), "l2": (250, 20), "l3": (200, 10)},
                     print_summary = False,
                     batch_train = False,
                     random_sequencing = True,
                     return_train_acts = False,
                     seed = 0,
                     activation = nn.LeakyReLU(0.1),
                     lr = 0.01):
        modelsMap = {}
        trainActivationsMap = {}
        testActivationsMap = {}
        seed = generate_seeds(seed, 1)
        torch.manual_seed(seed[0])
        for modelName, get_model in models.items():
            model = get_model(
                in_features = in_features, 
                out_features = out_features, 
                layers = layers, 
                model_params = model_params,
                activation = activation)

            if ortho_lambda and "Ortho" in modelName:
                _ , _ ,_ , _, _, _, trainActivations, testActivations = self.trainUtil.train_and_evaluate(model, 
                                                epochs = n_epochs, 
                                                name = modelName,
                                                print_msg = print_summary,
                                                batch_train = batch_train,
                                                ortho_lambda = ortho_lambda,
                                                random_sequencing = random_sequencing,
                                                return_train_acts = return_train_acts,
                                                lr = lr)

            else:
                _ , _ ,_ , _, _ , _, trainActivations, testActivations = self.trainUtil.train_and_evaluate(model, 
                                                epochs = n_epochs, 
                                                name = modelName,
                                                print_msg = print_summary,
                                                batch_train = batch_train,
                                                random_sequencing = random_sequencing,
                                                return_train_acts = return_train_acts,
                                                lr = lr)
                
            modelsMap[modelName] = model
            trainActivationsMap[modelName] = trainActivations
            testActivationsMap[modelName] = testActivations
        
        return modelsMap, trainActivationsMap, testActivationsMap