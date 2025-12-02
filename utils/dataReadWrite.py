import os

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
            df_train = pd.DataFrame({
                "epoch": epochs,
                "mean_acc": np.array(mean_acc),
                "bt_l": np.array(bt_ls),
                "bt_u": np.array(bt_us),
            })
            df_thr_per_rn = pd.DataFrame({
                "epochs_to_90": epochs_to_threshold[90.],
                "epochs_to_95": epochs_to_threshold[95.],
            })
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