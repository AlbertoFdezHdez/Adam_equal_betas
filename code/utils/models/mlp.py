import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_dim):
        super(MLP, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Flatten())
        
        # Compute the flatten output shape for the classification module
        with torch.no_grad():
            random_input = torch.rand(size=(1, *input_shape))
            out = self.feature_extraction(random_input)
            input_shape = out.shape[-1]
        
        self.classification = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes))
        
        
    def forward(self, x):
        x = self.feature_extraction(x)
        out = self.classification(x)
        return out
    
class MLP_doubleReLU(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_dim):
        super(MLP_doubleReLU, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Flatten())
        
        # Compute the flatten output shape for the classification module
        with torch.no_grad():
            random_input = torch.rand(size=(1, *input_shape))
            out = self.feature_extraction(random_input)
            input_shape = out.shape[-1]
        
        self.layers = nn.ModuleList([
            nn.Linear(input_shape, hidden_dim),
            nn.Linear(hidden_dim, num_classes),
            nn.Linear(hidden_dim, num_classes)
        ])
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.layers[0](x)
        x1 = self.activation(x)
        x2 = self.activation(-1 * x)
        x1 = self.layers[1](x1)
        x2 = self.layers[2](x2)
        out = x1+x2
        return out