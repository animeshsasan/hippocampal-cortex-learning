import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from models.enums.chartType import TrainingCharts
from models.trainingAnalysis.experimentResults import ExperimentResults
from models.trainingAnalysis.modelTrainingData import ModelTrainingData

class StoreData():
    output_folder = "../outputs/"

    def store_df(self, df, file_path):
        full_path = os.path.join(self.output_folder, file_path)

        # Extract directory part
        dir_path = os.path.dirname(full_path)

        # Create directories if they do not exist
        if dir_path != "":
            os.makedirs(dir_path, exist_ok=True)

        # Add .csv extension
        csv_path = full_path + ".csv"
        
        df.to_csv(csv_path, index=False)

    def pad_to_same_length(self, a, b):
        max_len = max(len(a), len(b))
        a = list(a) + [None] * (max_len - len(a))
        b = list(b) + [None] * (max_len - len(b))
        return a, b

    def store_experiment_data(
            self,
            experiment: ExperimentResults,
            file_path: str
        ):

        for modelName, modelData in experiment.train_data.items():
            mean_acc, bt_ls, bt_us, epochs = modelData.get_mean_bt_acc()
            epochs_to_threshold = modelData.get_epochs_to_thr(modelName, [90., 95.])
            
            final_train_mean, final_train_bt_l, final_train_bt_h = experiment.test_data[modelName].get_train_mean_bootstrapped()
            final_val_mean, final_val_bt_l, final_val_bt_h = experiment.test_data[modelName].get_val_mean_bootstrapped()
            
            # Build a DF for this model
            # print(modelName, len(epochs), len(mean_acc), len(bt_ls), len(bt_us))
            df_train = pd.DataFrame({
                "epoch": epochs,
                "mean_acc": np.array(mean_acc),
                "bt_l": np.array(bt_ls),
                "bt_u": np.array(bt_us),
            })
            e90, e95 = self.pad_to_same_length(
                epochs_to_threshold[90.],
                epochs_to_threshold[95.]
            )

            df_thr_per_rn = pd.DataFrame({
                "epochs_to_90": e90,
                "epochs_to_95": e95,
            })
            # print(len(final_train_mean), len(final_train_bt_l), len(final_train_bt_h), len(final_val_mean), len(final_val_bt_l), len(final_val_bt_h))
            df_test = pd.DataFrame({
                "final_train_mean": final_train_mean,
                "final_train_bt_l": final_train_bt_l,
                "final_train_bt_h": final_train_bt_h,
                "final_val_mean": final_val_mean,
                "final_val_bt_l": final_val_bt_l,
                "final_val_bt_h": final_val_bt_h,
            }, index=[0])
            
            self.store_df(df_train, file_path + f"/train/{modelName}")
            self.store_df(df_thr_per_rn, file_path + f"/train/epochs_to_thr/{modelName}")
            self.store_df(df_test, file_path + f"/test/{modelName}")

    def store_train_similarity_complete_multi_run(
        self,
        similarity_data: Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]],
        file_path: str,
        needBootstrapped: bool = False
    ):
        """
        Stores similarity data from create_train_similarity_complete_df_multi_run.

        Structure:
        similarity_data = {
            model_name: {
                layer_name: (mean_df, bt_df)
            }
        }
        """

        for model_name, layer_dict in similarity_data.items():
            for layer_name, (df_mean, df_bt) in layer_dict.items():

                # Store mean similarity DF
                self.store_df(
                    df_mean,
                    f"{file_path}/train_similarity/{model_name}/{layer_name}/mean"
                )

                # Store bootstrapped DF if needed
                if needBootstrapped and df_bt is not None:
                    self.store_df(
                        df_bt,
                        f"{file_path}/train_similarity/{model_name}/{layer_name}/bootstrapped"
                    )