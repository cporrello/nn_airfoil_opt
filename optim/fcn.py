import torch.nn as nn

class FullyConnectedNetwork(nn.Module):
    
    # Initialize class object
    def __init__(self, input_dim, output_dim, n_layers, n_units, activation):
        super(FullyConnectedNetwork, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._n_layers = n_layers
        self._n_units = n_units
        self._activation = activation

        # Create fully connected DNN with variable number of hidden layers
        fcn_modules = nn.ModuleList()
        fcn_modules.append(nn.Linear(self._input_dim, self._n_units))
        for hidden_layer in range(self._n_layers-2):
            fcn_modules.append(nn.Linear(self._n_units, self._n_units))
            fcn_modules.append(self._activation)
        fcn_modules.append(nn.Linear(self._n_units, self._output_dim))
        self.fcn = nn.Sequential(*fcn_modules)

    # Forward propagation
    def forward(self, x):
        return self.fcn(x)