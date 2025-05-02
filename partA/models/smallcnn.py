import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# Activation function map
ACT_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish
}

class SmallCNN(nn.Module):
    def __init__(self, filters = [16,32,64,32,16],  activation = 'relu', kernel_size = 3,  dense_neurons = 100, num_layers=5, in_channels=3, num_classes = 10):
        super().__init__()
        # Accessing the parsed arguments
        self.num_layers = num_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation_fn = ACT_MAP[activation.lower()]
        self.dense_neurons = dense_neurons
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Check that the length of filters matches the number of layers
        assert len(self.filters) == self.num_layers, "Length of filters must match number of layers"

        self.conv_blocks = nn.ModuleList()
        current_channels = self.in_channels
        
        for i in range(self.num_layers):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, self.filters[i], kernel_size=self.kernel_size, padding=self.kernel_size // 2),
                    self.activation_fn(),
                    nn.MaxPool2d(2)
                )
            )
            current_channels = self.filters[i]  # Set the current number of channels to the filters in this layer

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(current_channels*8*8, self.dense_neurons)
        self.fc2 = nn.Linear(self.dense_neurons, self.num_classes)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.flatten(x)  # (B, C)
        x = self.fc1(x)
        x = self.activation_fn()(x)
        x = self.fc2(x)
        return x

# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser(description='CNN model configuration')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of convolutional layers')
    parser.add_argument('--filters', type=int, nargs='+', default=[16, 32, 64, 32, 16], 
                        help='List of number of filters per convolutional layer')
    parser.add_argument('--kernel_size', type=int, default=3, help='Size of convolution kernels')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leakyrelu', 'sigmoid', 'tanh'],
                        help='Activation function for convolution layers')
    parser.add_argument('--dense_neurons', type=int, default=512, help='Number of neurons in the dense layer')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes (default: 10)')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels (e.g., RGB: 3)')
    
    return parser.parse_args()

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    args = vars(args) # convert to dictionary
    
    # Initialize the model with parsed arguments
    model = SmallCNN(args)

    # Optionally print the model
    print(model)
    x = torch.randn(16,3,256,256)
    x.shape