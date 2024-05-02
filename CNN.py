import torch
import torch.nn as nn
from MLP import MLP


class CNN(nn.Module):
    def __init__(self,
                 input_size,
                 num_conv_layers,
                 conv_channels,
                 conv_kernel_size,
                 conv_kernel_stride,
                 conv_kernel_padding,
                 conv_activation,
                 conv_pooling,
                 conv_pooling_kernel_size,
                 conv_pooling_stride,
                 conv_normalization,
                 hidden_dims,
                 output_dim,
                 mlp_activation,
                 mlp_normalization,
                 mlp_dropout_rate,
                 device):
        """
        Initialize the CNN model.

        Parameters:
            input_size (int): Input size of features.
            num_conv_layers (int): Number of convolutional layers.
            conv_channels (list of ints): Number of channels for each convolutional layer.
            conv_kernel_size (list of ints): Kernel size for each convolutional layer.
            conv_kernel_stride (list of ints): Stride for each convolutional layer.
            conv_kernel_padding (list of ints): Padding for each convolutional layer.
            conv_activation (torch.nn.Module): Activation function for convolutional layers.
            conv_pooling (torch.nn.Module): Pooling function for convolutional layers.
            conv_pooling_kernel_size (list of ints): Kernel size for pooling for each convolutional layer.
            conv_pooling_stride (list of ints): Stride for pooling for each convolutional layer.
            conv_normalization (torch.nn.Module): Normalization function for convolutional layers.
            hidden_dims (list of ints): Dimensions of hidden layers in the MLP.
            output_dim (int): Output dimension of the model.
            mlp_activation (torch.nn.Module): Activation function for the MLP.
            mlp_normalization (torch.nn.Module): Normalization function for the MLP.
            mlp_dropout_rate (float): Dropout rate for the MLP.
            device (torch.device): Device on which the model will be run.
        """
        super(CNN, self).__init__()

        # Define input size of features
        self.input_size = input_size

        # Establish the Lists to save the later layers in
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # Define parameters for convolutional layers
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_kernel_stride = conv_kernel_stride
        self.conv_kernel_padding = conv_kernel_padding

        # Define parameters for the convolution activation
        self.conv_activation = conv_activation

        # Define parameters for the pooling function
        self.conv_pooling = conv_pooling
        self.conv_pooling_kernel_size = conv_pooling_kernel_size
        self.conv_pooling_stride = conv_pooling_stride

        # Define parameters for the normalization
        self.conv_normalization = conv_normalization

        # Define parameters for the MLP
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.mlp_activation = mlp_activation
        self.mlp_normalization = mlp_normalization
        self.mlp_dropout_rate = mlp_dropout_rate

        # Device to run the Net on, helps potential tensor creation
        self.device = device

        # Define the convolutional blocks
        for layer in range(self.num_conv_layers):
            # Append the convolutional layer
            self.conv_layers.append(nn.Conv1d(self.conv_channels[layer],
                                              self.conv_channels[layer+1],
                                              kernel_size=self.conv_kernel_size[layer],
                                              stride=self.conv_kernel_stride[layer],
                                              padding=self.conv_kernel_padding[layer]))

            # Append the Pooling layer
            self.pool_layers.append(self.conv_pooling(kernel_size=self.conv_pooling_kernel_size[layer],
                                                      stride=self.conv_pooling_stride[layer]))

            # Append the normalization layer
            self.bn_layers.append(self.conv_normalization(self.conv_channels[layer+1]))

        # Calculate flattened shape for MLP
        self.C_out, self.L_out = self.calc_conv_output_size()

        # MLP layers
        self.mlp = MLP(input_dim=self.C_out*self.L_out,
                       hidden_dims=self.hidden_dims,
                       output_dim=self.output_dim,
                       activation=self.mlp_activation,
                       normalization=self.mlp_normalization,
                       dropout_rate=self.mlp_dropout_rate).to(self.device).apply(self.init_weights)

    def init_weights(self, m):
        """
        Initialize weights for the fully connected layers of the MLP.

        Parameters:
            m (torch.nn.Module): Module to initialize weights for.
        """
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def calc_conv_output_size(self):
        """
        Calculate the output size after convolutional layers.

        Returns:
            tuple: Number of output channels and length of output feature map.
        """
        L_in = self.input_size
        L_out = None
        dilation = 1  # parameter standard, not otherwise defined in CNN as of yet
        pool_padding = 0  # parameter standard, not otherwise defined in CNN as of yet
        for layer in range(self.num_conv_layers):
            # Convolution
            L_out = (L_in + 2 * self.conv_kernel_padding[layer] - dilation * (self.conv_kernel_size[layer] - 1) - 1) / \
                    self.conv_kernel_stride[layer] + 1

            # Set L_in for the next layer
            L_in = L_out

            # Pooling
            L_out = (L_in + 2 * pool_padding - dilation * (self.conv_pooling_kernel_size[layer] - 1) - 1) / \
                self.conv_pooling_stride[layer] + 1

            # Set L_in for the next layer
            L_in = L_out

        return int(self.conv_channels[-1]), int(L_out)

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Reshape the matrix input to (N, C_in, L_in)
        x = torch.unsqueeze(x, dim=1)

        # Convolutional layers with activation, batch normalization and pooling
        for conv, pool, bn in zip(self.conv_layers, self.pool_layers, self.bn_layers):
            # Apply convolution
            x = conv(x)
            # Apply activation
            x = self.conv_activation(x)
            # Apply normalization
            x = bn(x)
            # Apply pooling
            x = pool(x)

        # Reshape the tensor for the fully connected layer
        x = x.view(-1, self.C_out * self.L_out)

        # Fully connected MLP layer
        x = self.mlp(x)
        return x
