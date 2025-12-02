import pandas as pd
import torch

class DummyDataset():
    _COLUMNS = ["a", "b", "c", "d", "e", "f", "g", "category", "type", "ttsplit"]
    _FEATURE_COLS = ["a", "b", "c", "d", "e", "f", "g"]
    _TARGET_COL = "category"
    _SPLIT_COL = "ttsplit"

    def get_raw_data(self):
        data = [
            [1,1,0,0,1,1,0,1,"standard","train"],
            [0,1,1,1,0,1,0,1,"standard","test"],
            [1,0,1,0,1,1,0,1,"standard","test"],
            [1,1,0,1,1,0,0,1,"standard","test"],
            [0,0,1,0,0,0,1,1,"exception","train"],
            [0,0,0,1,0,0,1,1,"exception","train"],
            [0,0,1,1,0,0,0,2,"standard","train"],
            [0,1,0,0,0,1,0,2,"standard","test"],
            [0,0,1,0,1,0,0,2,"standard","test"],
            [1,0,0,1,0,0,0,2,"standard","test"],
            [1,1,1,0,1,1,1,2,"exception","train"],
            [1,1,0,1,1,1,1,2,"exception","train"],
            [1,1,1,1,1,1,0,1,"prototype","train"],
            [0,0,1,1,1,1,0,1,"standard","train"],
            [1,1,0,1,0,1,0,1,"standard","train"],
            [1,0,1,1,1,0,0,1,"standard","train"],
            [0,1,1,0,1,1,0,1,"standard","train"],
            [1,0,0,0,0,0,1,1,"exception","train"],
            [0,1,0,0,0,0,1,1,"exception","train"],
            [0,0,0,0,0,0,0,2,"prototype","train"],
            [0,0,0,0,1,1,0,2,"standard","train"],
            [0,1,0,1,0,0,0,2,"standard","train"],
            [1,0,0,0,1,0,0,2,"standard","train"],
            [0,0,1,0,0,1,0,2,"standard","train"],
            [1,1,1,1,1,0,1,2,"exception","train"],
            [1,1,1,1,0,1,1,2,"exception","train"]
            ]


        df = pd.DataFrame(data, columns=self._COLUMNS)
        return df

    def get_original_split(self):
        s = "standard"
        e = "exception"
        p = "prototype"
        t = "test"
        l = "train"
        data = [
            [1,1,0,0,1,1,0,1,s,t],
            [0,1,1,1,0,1,0,1,s,t],
            [1,0,1,0,1,1,0,1,s,t],
            [1,1,0,1,1,0,0,1,s,t],
            [0,0,1,0,0,0,1,1,e,t],
            [0,0,0,1,0,0,1,1,e,t],
            [0,0,1,1,0,0,0,2,s,t],
            [0,1,0,0,0,1,0,2,s,t],
            [0,0,1,0,1,0,0,2,s,t],
            [1,0,0,1,0,0,0,2,s,t],
            [1,1,1,0,1,1,1,2,e,t],
            [1,1,0,1,1,1,1,2,e,t],
            [1,1,1,1,1,1,0,1,p,l],
            [0,0,1,1,1,1,0,1,s,l],
            [1,1,0,1,0,1,0,1,s,l],
            [1,0,1,1,1,0,0,1,s,l],
            [0,1,1,0,1,1,0,1,s,l],
            [1,0,0,0,0,0,1,1,e,l],
            [0,1,0,0,0,0,1,1,e,l],
            [0,0,0,0,0,0,0,2,p,l],
            [0,0,0,0,1,1,0,2,s,l],
            [0,1,0,1,0,0,0,2,s,l],
            [1,0,0,0,1,0,0,2,s,l],
            [0,0,1,0,0,1,0,2,s,l],
            [1,1,1,1,1,0,1,2,e,l],
            [1,1,1,1,0,1,1,2,e,l],
            ]
        df = pd.DataFrame(data, columns=self._COLUMNS)
        return df
    
    def get_dataset(self, original_split= False, unique = True):
        if original_split:
            df = self.get_original_split()
        else:
            df = self.get_raw_data()

        if unique:
            train_df = df[df[self._SPLIT_COL] == "train"]
            test_df = df[df[self._SPLIT_COL] == "test"]

            X_train = torch.tensor(train_df[self._FEATURE_COLS].to_numpy(), dtype=torch.float32)
            X_val = torch.tensor(test_df[self._FEATURE_COLS].to_numpy(), dtype=torch.float32)

            y_train = torch.tensor((train_df[self._TARGET_COL] - 1).to_numpy(), dtype=torch.int64)
            y_val = torch.tensor((test_df[self._TARGET_COL] - 1).to_numpy(), dtype=torch.int64)

            return X_train, y_train, X_val, y_val
        else:
            X = torch.tensor(df[self._FEATURE_COLS].to_numpy(), dtype=torch.float32)
            y = torch.tensor(df[[self._TARGET_COL]].to_numpy(), dtype=torch.int64)
            
            return X, y, X, y