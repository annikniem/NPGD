#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and validate

This file contains the main program that can be used to train, validate and
evaluate the NPGD model. It saves plots of the training and validation losses
as well as images of the final results.
"""

# %% Imports
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import itertools

import data_loader as AMRI_dataloader
import model as NPGD_model
import helper_functions as helpers

# %% Loading data
data_loc = 'NPGD/data' # Use a location that works for you
os.makedirs(data_loc, exist_ok=True)
batch_size = 32

# Creating dataloaders for training, testing and validation
train_loader, test_loader, val_loader = AMRI_dataloader.create_dataloaders(data_loc, batch_size)

# %% Training and validating NPGD
epochs = 10
input_size = 1
hidden_size = 64
output_size = 1
num_layers = 5
learning_rate = 0.001
mu_learning_rate = 0.05  # Separate learning rate for mu

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, and optimizer
model = NPGD_model.NPGD(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()

# Create parameter groups for the optimizer
optimizer = optim.Adam([
    {'params': model.mus, 'lr': mu_learning_rate},
    {'params': model.conv1.parameters(), 'lr': learning_rate},
    {'params': model.conv2.parameters(), 'lr': learning_rate},
    {'params': model.conv3.parameters(), 'lr': learning_rate},
    {'params': model.conv4.parameters(), 'lr': learning_rate}
], lr=learning_rate)

# Print the number of parameters
print(f'Total number of trainable parameters: {helpers.count_parameters(model)}')

train_losses = []
val_losses = []

# Training and validation loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for kspace, M, gt in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
        kspace, M, gt = kspace.to(device), M.to(device), gt.to(device)
        optimizer.zero_grad()

        k_space = helpers.calculate_k_space(gt)
        part_space = helpers.apply_mask(k_space, M)
        reconstructed = helpers.inverse_fft(part_space)

        outputs = model(torch.abs(reconstructed).to(device), M)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer.step()
        
        helpers.clamp_mus(model)

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Evaluate on test set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for kspace, M, gt in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} (Validation)'):
            kspace, M, gt = kspace.to(device), M.to(device), gt.to(device)

            k_space = helpers.calculate_k_space(gt)
            part_space = helpers.apply_mask(k_space, M)
            reconstructed = helpers.inverse_fft(part_space)

            outputs = model(torch.abs(reconstructed).to(device), M)
            loss = criterion(outputs, gt)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# %% Plotting and saving the training and validation losses
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.tight_layout()
plt.savefig('NPGD_losses.png', dpi=300, bbox_inches='tight')
plt.show()

# %% Plotting and saving the results
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
num_iterations = 5

with torch.no_grad():
    for i, (kspace, M, gt) in enumerate(itertools.islice(test_loader, num_iterations)):
        kspace, M, gt = kspace.to(device), M.to(device), gt.to(device)
        part_space = helpers.apply_mask(kspace, M)
        reconstructed = helpers.inverse_fft(part_space)

        axes[0, i].imshow(reconstructed[0, :, :].cpu(), vmin=-2.3, interpolation='nearest', cmap='gray')
        axes[0, i].set_title('MRI image (k-space reconst.)')
        axes[0, i].axis('off')

        MRI_CNN_reconstructed = model(torch.abs(reconstructed).to(device), M)

        axes[1, i].imshow(MRI_CNN_reconstructed[0, :, :].cpu(), vmin=-2.3, interpolation='nearest', cmap='gray')
        axes[1, i].set_title('MRI image (NPGD reconst.)')
        axes[1, i].axis('off')

        axes[2, i].imshow(gt[0, :, :].cpu(), vmin=-2.3, interpolation='nearest', cmap='gray')
        axes[2, i].set_title('MRI image (ground truth)')
        axes[2, i].axis('off')

plt.tight_layout()
plt.savefig('NPGD_results.png', dpi=300, bbox_inches='tight')
plt.show()

# %% Calculating the MSE of the entire test set
total_test_mse = 0
num_batches = len(test_loader)

with torch.no_grad():
    for kspace, M, gt in tqdm(test_loader):
        kspace, M, gt = kspace.to(device), M.to(device), gt.to(device)

        k_space = helpers.calculate_k_space(gt)
        part_space = helpers.apply_mask(k_space, M)
        reconstructed = helpers.inverse_fft(part_space)

        outputs = model(torch.abs(reconstructed).to(device), M)
        loss = criterion(outputs, gt)
        total_test_mse += loss.item()

total_test_mse /= num_batches
print(f'Test MSE: {total_test_mse:.4f}')