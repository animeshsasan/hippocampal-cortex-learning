import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P
from torch.nn.functional import linear
from torch import Tensor
import math


class CustomLayer(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                out_features_allowed = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 

        self.bias = nn.Parameter(torch.empty((out_features)))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.out_features_allowed = out_features_allowed
        self.activation = nn.LeakyReLU(0.1)
        

    def forward(self, input: Tensor) -> Tensor:
        """
        input: (batch_size, in_features)
        """
        outputs = linear(input, self.weight, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if self.out_features_allowed is not None and self.out_features_allowed < self.out_features:
            topk_values, indices = torch.topk(outputs, self.out_features_allowed, dim=1)
        
            # Create output tensor filled with zeros
            topk_outputs = torch.zeros_like(outputs).scatter(1, indices, topk_values)
        else:
            topk_outputs = outputs
  
        return topk_outputs
