from typing import Callable
from layers.customLayer import CustomLayer
import torch.nn as nn

class ModelSetups():

    def _get_sparse_model(self, 
                          in_features = 2,
                          out_features = 2, 
                          layers = 3, 
                          model_params: dict[str, tuple[int, int | None]] = {"l1": (200, 15), "l2": (250, 20), "l3": (200, 10)},
                          activation = nn.LeakyReLU(0.1)) -> nn.Module:
        assert layers == len(model_params.keys()), "Number of layers must match the length of model_params"
        class SparseModel(nn.Module):
            def __init__(self, in_features=in_features, out_features=out_features):
                super().__init__()
                self.layers = nn.ModuleList()
                self.layer_names = list()

                # First layer: input → first hidden
                prev_out = in_features
                for layer_name, (hidden_size, out_allowed) in model_params.items():
                    self.layers.append(CustomLayer(prev_out, hidden_size, out_features_allowed=out_allowed, activation = activation))
                    prev_out = hidden_size
                    self.layer_names.append(layer_name)

                # Final layer: last hidden → output
                self.layers.append(CustomLayer(prev_out, out_features, activation = activation))

            def forward(self, x, return_acts=False):
                activations = {}
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if return_acts:
                        layer_name = self.layer_names[i] if i < len(self.layer_names) else f"out"
                        activations[layer_name] = x
                return (x, activations) if return_acts else x

        return SparseModel()
    
    def _get_control_model(self, 
                           in_features = 2, 
                           out_features = 2, 
                           layers = 3, 
                           model_params: dict[str, tuple[int, int | None]] = {"l1": (200, 15), "l2": (250, 20), "l3": (200, 10)},
                           activation = nn.LeakyReLU(0.1)) -> nn.Module:
        assert layers == len(model_params.keys()), "Number of layers must match the length of model_params"
        class ControlModel(nn.Module):
            def __init__(self, in_features=in_features, out_features=out_features):
                super().__init__()
                self.layers = nn.ModuleList()
                self.layer_names = list()

                # First layer: input → first hidden
                prev_out = in_features
                for layer_name, (_, out_allowed) in model_params.items():
                    self.layers.append(CustomLayer(prev_out, out_allowed, activation = activation))
                    prev_out = out_allowed
                    self.layer_names.append(layer_name)

                # Final layer: last hidden → output
                self.layers.append(CustomLayer(prev_out, out_features, activation = activation))

            def forward(self, x, return_acts=False):
                activations = {}
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if return_acts:
                        layer_name = self.layer_names[i] if i < len(self.layer_names) else f"out"
                        activations[layer_name] = x
                return (x, activations) if return_acts else x

        return ControlModel()
    
    def _get_dense_model(self, 
                         in_features = 2, 
                         out_features = 2, 
                         layers = 3, 
                         model_params: dict[str, tuple[int, int | None]] = {"l1": (200, 15), "l2": (250, 20), "l3": (200, 10)},
                         activation = nn.LeakyReLU(0.1)) -> nn.Module:
        assert layers == len(model_params.keys()), "Number of layers must match the length of model_params"
        class DenseModel(nn.Module):
            def __init__(self, in_features=in_features, out_features=out_features):
                super().__init__()
                self.layers = nn.ModuleList()
                self.layer_names = list()

                # First layer: input → first hidden
                prev_out = in_features
                for layer_name, (hidden_size, _) in model_params.items():
                    self.layers.append(CustomLayer(prev_out, hidden_size, activation = activation))
                    prev_out = hidden_size
                    self.layer_names.append(layer_name)

                # Final layer: last hidden → output
                self.layers.append(CustomLayer(prev_out, out_features, activation = activation))

            def forward(self, x, return_acts=False):
                activations = {}
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if return_acts:
                        layer_name = self.layer_names[i] if i < len(self.layer_names) else f"out"
                        activations[layer_name] = x
                return (x, activations) if return_acts else x

        return DenseModel()


    def get_all_models(self, 
                    model_type: str = "all") -> dict[str, Callable]:
            
            
        if model_type == "sparse":
            return {"Sparse Model": self._get_sparse_model}
        elif model_type == "control":
            return {"Control Model": self._get_control_model}
        elif model_type == "dense":
            return {"Dense Model": self._get_dense_model}
        else:
            return {
                "Sparse Model": self._get_sparse_model,
                "Control Model": self._get_control_model,
                "Dense Model": self._get_dense_model
            }