import torch.nn as nn
from torch import Tensor

from layers.customLayer import CustomLayer
from layers.customLinearLayer import CustomLinearLayer


class HippocampusLayer(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                out_features_allowed,
                activation = nn.LeakyReLU(0.1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_allowed = out_features_allowed

        assert self.out_features >= 5 * self.out_features_allowed, "Alpha is too small for the given out_features. \
            Increase the number of out_features or reduce the out_features_allowed"

        self.dg_cells = CustomLayer(self.in_features, self.out_features, self.out_features_allowed)

        self.g_cell_inhibition = activation
        

    def forward(self, input: Tensor) -> Tensor:
        """
        input: (batch_size, in_features)
        """
        outputs = self.dg_cells(input)
        if self.g_cell_inhibition is not None:
            outputs = self.g_cell_inhibition(outputs)
        
        return outputs
        
