#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPGD model

This file contains the Neural Proximal Gradient Descent (NPGD) model architecture.
NPGD can be used to reconstruct accelerated MRI images. The NPGD model
introduces mu as a learnable parameter and uses several convolutional layers
to perform the proximal step. This way state-of-the-art Accelerated MRI
reconstruction can be performed.

Model Structure
   - The model consists of several unfolded layers defined by num_layers
   - Each unfolded layer consists of a gradient step and proximal step
   - A learnable paraemeter mu is introduced for gradient computation
   - The proximal step is learned using 4 convolutional layers
   - ReLU activation function is used to introduce non-linearity

"""

# %% Imports
import torch
import torch.nn as nn

import helper_functions as helpers

# %% NPGD model architecture
class NPGD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(NPGD, self).__init__()
        self.num_layers = num_layers

        # Define multiple layers for the proximal steps
        self.conv1 = nn.ModuleList([nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1) for _ in range(num_layers)])
        self.conv2 = nn.ModuleList([nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1) for _ in range(num_layers)])
        self.conv3 = nn.ModuleList([nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1) for _ in range(num_layers)])
        self.conv4 = nn.ModuleList([nn.Conv2d(hidden_size, output_size, kernel_size=3, padding=1) for _ in range(num_layers)])
        
        # Define multiple mus
        self.mus = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)])
        
        # Activation function
        self.relu = nn.ReLU()
    
    # Define unfolded layer
    def layer_run(self, x, z, i, M):
        # Gradient step
        grad = helpers.calculate_k_space(z) - self.mus[i] * M * helpers.calculate_k_space(z) + self.mus[i] * M * helpers.calculate_k_space(x)
        z = helpers.inverse_fft(grad)
        z = z.unsqueeze(dim=1)
        # Proximal step
        z = self.relu(self.conv1[i](z))                              
        z = self.relu(self.conv2[i](z))                                
        z = self.relu(self.conv3[i](z))                                
        z = self.relu(self.conv4[i](z))                                
        return z.squeeze()

    def forward(self, x, M):
        # Initialize solution z
        z = torch.zeros_like(x)
        # Adding unfolded layers to model
        for i in range(self.num_layers):  
            z = self.layer_run(x, z, i, M)
        return z