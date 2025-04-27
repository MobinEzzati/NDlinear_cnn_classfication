from socket import send_fds

import torch
import torch.nn as nn
from NdLinear.ndlinear import NdLinear

import math

class NdLinearCNN(nn.Module):
    """
    CNN model with Conv layers + NdLinear layer for image classification on SVHN.
    """

    def __init__(self, input_shape=(64, 8, 8), hidden_shape=(32, 8, 8)):
        super(NdLinearCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.ndlinear = NdLinear(input_shape, hidden_shape)

        final_dim = math.prod(hidden_shape)  # flatten size after NdLinear
        self.fc_out = nn.Linear(final_dim, 10)  # SVHN has 10 classes (digits 0-9)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.ndlinear(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc_out(self.relu(x))
        return x

