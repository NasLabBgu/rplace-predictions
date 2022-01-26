# Authors: Abraham Israeli
# Python version: 3.7
# Last update: 26.01.2021

from neural_net.nn_classifier import NNClassifier
import torch
import torch.nn.functional as F
from skorch import NeuralNetClassifier


class MyModule(torch.nn.Module):
    def __init__(self, input_dim, hid_size=100, dropout_perc=0.5, nonlin=F.relu):
        super(MyModule, self).__init__()
        self.dense0 = torch.nn.Linear(input_dim, hid_size)
        self.nonlin = nonlin
        self.dropout = torch.nn.Dropout(dropout_perc)
        self.dense1 = torch.nn.Linear(hid_size, 10)
        self.output = torch.nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(input=self.output(X))
        return X


class BertNNModel(NNClassifier):
    def __init__(self, input_dim, model, eval_measures, epochs=10, hid_size=100, dropout_perc=0.5,
                 nonlin=F.relu, layers_amount=2, use_meta_features=True, seed=1984):
        super(BertNNModel, self).__init__(model=model, eval_measures=eval_measures, hid_size=hid_size,
                                          epochs=epochs, use_meta_features=use_meta_features, seed=seed)
        self.input_dim = input_dim
        self.my_module = MyModule(input_dim=self.input_dim, hid_size=hid_size, dropout_perc=dropout_perc)
        self.layers_amount = layers_amount

    def build_model(self, lr=0.1, suffle_train=True):
        net = NeuralNetClassifier(self.my_module, max_epochs=self.epochs, lr=lr,
                                  iterator_train__shuffle=suffle_train)
        return net




