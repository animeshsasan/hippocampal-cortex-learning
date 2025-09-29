import torch.nn as nn
import torch.nn.utils.parametrizations as P
from torch.nn.functional import linear
from torch import Tensor

from layers.customLinearLayer import CustomLinearLayer


class NeoCortexLayer(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                hippocampol_connections,
                memory_size,
                activation = nn.LeakyReLU(0.1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.memory_size = memory_size
        self.hippocampol_connections = hippocampol_connections

        # According to Marrs Model cortical Pyramids is 10^8 and 10^6 indicator cells
        assert memory_size >= 20 * out_features, "Memory size should be at least 100 times the out_features"

        self.coritcal_input = CustomLinearLayer(self.in_features, self.hippocampol_connections)
        self.coritcal_memory = CustomLinearLayer(self.hippocampol_connections, self.memory_size)
        self.indicator_cells = CustomLinearLayer(self.memory_size, self.out_features)

        self.s_cell_inhibition = activation
        

    def forward(self, input: Tensor) -> Tensor:
        """
        input: (batch_size, in_features)
        """
        filtered_input = self.coritcal_input(input)
        memory_representation = self.coritcal_memory(filtered_input)
        outputs = self.indicator_cells(memory_representation)

        if self.s_cell_inhibition is not None:
            outputs = self.s_cell_inhibition(outputs)
  
        return outputs
        
