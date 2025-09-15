import torch
import torch.nn as nn
from torch.nn.functional import linear, softmax
from torch import Tensor


class SparseCodedLayer(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                out_features_allowed = None,
                activation = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.randn((out_features, in_features)) * 0.01
        )
        self.bias = nn.Parameter(
            torch.randn((out_features)) * 0.01
        )
        self.out_features_allowed = out_features_allowed
        self.activation = activation
        

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
