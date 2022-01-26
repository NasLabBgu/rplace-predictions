# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 06.08.2019

import torch.nn as nn


class PytorchMLP(nn.Module):
    """
    class to handle and run simple multi-layer perceptron (MLP) using pytorch

    Parameters
    ----------
    num_features: int
        number of features of each instance in the dataset
    dropout: float, default: 0.25
        standrad droput parameter to be used in the modeling phase
    n_hid: int, default: 128
        number of hidden nodes in each hidden layers (number of neurons in each hidden layer)

    Attributes
    ----------
    model: pytorch model object
        the model which being used
    """
    def __init__(self, num_features, dropout=0.25, n_hid=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, n_hid),
            nn.ReLU(),
            #nn.BatchNorm1d(n_hid),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid // 4),
            nn.ReLU(),
            #nn.BatchNorm1d(n_hid // 4),
            nn.Dropout(dropout),
            nn.Linear(n_hid // 4, 2),
            nn.Softmax()
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)
