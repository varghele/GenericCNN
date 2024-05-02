import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU(), normalization=None, dropout_rate=0.0):
        """
        Initialize the Multi-Layer Perceptron (MLP) model.

        Parameters:
            input_dim (int): Dimensionality of the input features.
            hidden_dims (list of ints): List containing the sizes of hidden layers.
            output_dim (int): Dimensionality of the output.
            activation (torch.nn.Module): Activation function to be applied between hidden layers.
            normalization (torch.nn.Module): Normalization function to be applied between hidden layers.
            dropout_rate (float): Dropout rate to be applied between hidden layers.
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.normalization = normalization

        # Create the layers
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            # Linear Layers
            layers.append(nn.Linear(in_dim, hidden_dim))
            # Activation
            layers.append(self.activation)
            # Normalization
            if self.normalization is not None:
                layers.append(self.normalization(hidden_dim))
            # Dropout Regularization
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.layers(x)
