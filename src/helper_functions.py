#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions

This file contains all helper functions needed for training and validating NPGD:
   - MRI helper functions are used for MRI data processing
   - Counting learnable parameters
   - Weight clamping for mus
"""
# %% Imports
import torch.fft as fft

# %% MRI helper functions
def calculate_k_space(mri_image):
    return fft.fftshift(fft.fftn(mri_image, dim=(-2, -1)), dim=(-2, -1))

def apply_mask(k_space, M):
    M = M.to(k_space.dtype)
    return k_space * M

def inverse_fft(partial_k_space):
    return fft.ifftn(fft.ifftshift(partial_k_space, dim=(-2, -1)), dim=(-2, -1)).real

# %% Count the number of learnable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %% Clamp mus between 0 and 1
def clamp_mus(model, min_val=0.0, max_val=1.0):
    for param in model.mus:
        param.data.clamp_(min_val, max_val)