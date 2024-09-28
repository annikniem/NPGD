#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accelerated MRI Data loader

This file contains the code for a custom PyTorch data set and custom data loaders.
This code loads and pre processes accelerated MRI data stored in .npz files.
Each file contains MRI data along with the corresponding 
mask and ground truth images. The ground truth iamge and MRI data each consist of 
five 2D slices.

Attributes:
    split (str): indicates whether to load test or train data
    data_loc (str): defines the data location
    file_names (list): a list of paths to all the data files
    
Methods:
    __len__(): returns the length of the data set
    __getitem__(idx): returns all data (kspace, mask, gt) corresponding to one slice

"""

# %% Imports
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import numpy as np

# %% Accelerated MRI custom dataset
class Acc_MRI(Dataset):
    # Initialization of the custom dataset
    def __init__(self, split, data_loc):
        # Save the input parameters
        self.split    = split 
        self.data_loc = data_loc
        
        # Get all the file names
        self.file_names = glob.glob(f"{data_loc}//{split}//*.npz")
    
    # Return the number of images in this dataset
    def __len__(self):
        return len(self.file_names) * 5  # Every file contains 5 slices
    
    # Create a method that retrieves a single item from the dataset
    def __getitem__(self, idx):
        file_name = self.file_names[idx // 5] # Get the correct file (idx=slice index)
        data = np.load(file_name)
        
        kspace = data['kspace']
        M = data['M']
        gt = data['gt']
        
        # Get a specific slice of the MRI data at index idx % 5 - > (0-4)
        kspace = kspace[idx % 5, :, :]
        gt = gt[idx % 5, :, :]
        
        return kspace, M, gt

# %% Create dataloaders for the Acc_MRI dataset
def create_dataloaders(data_loc, batch_size, val_split = 0.2):
    dataset_train = Acc_MRI("train", data_loc)
    dataset_test  = Acc_MRI("test" , data_loc)
    
    val_size = int(val_split * len(dataset_train))
    train_size = len(dataset_train) - val_size
    
    dataset_train, dataset_val = random_split(dataset_train, [train_size, val_size])
    
    Fast_MRI_train_loader =  DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    Fast_MRI_test_loader  =  DataLoader(dataset_test , batch_size=batch_size, shuffle=True)
    Fast_MRI_val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    
    return Fast_MRI_train_loader, Fast_MRI_test_loader, Fast_MRI_val_loader

# %% Test if the dataloaders work
if __name__ == "__main__":
    # Define parameters
    data_loc = 'NPGD/data' # Use a location that works for you
    batch_size = 4
    
    train_loader, test_loader, val_loader = create_dataloaders(data_loc, batch_size)