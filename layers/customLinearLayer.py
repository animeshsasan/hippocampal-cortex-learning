import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P
from torch.nn.functional import linear
from torch import Tensor
import math


class CustomLinearLayer(nn.Module):
    def __init__(self, 
                in_features, 
                out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 

        self.bias = nn.Parameter(torch.empty((out_features)))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        

    def forward(self, input: Tensor) -> Tensor:
        """
        input: (batch_size, in_features)
        """
        
        return linear(input, self.weight, self.bias)
