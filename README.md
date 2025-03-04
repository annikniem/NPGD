# Neural Proximal Gradient Descent (NPGD) for Accelerated MRI reconstruction
In this repository you can find a PyTorch implementation of the NPGD algorithm. 
NPGD is a method for solving the ill-posed linear inverse problem of recovering images from highly compressed measurements. 
The algorithm is based on deep learning with embedded signal priors. The model architecture consists of several
unfolded layers, where each layer consists of a gradient step and a proximal step. 
NPGD can be used to achieve near state-of-the-art results in accelerated MRI reconstruction. [1]

## Repository contents
This repository contains the following:
- Src:
     - NPGD model architecture.
     - All  code required to train, validate and evaluate the NPGD model.
     - A custom dataloader for the MRI data.
- Figures:
     - Figure depicting the final results of MRI reconstruction using the NPGD model.
- Data:
     - Test and train data consisting of accelerated MRI measurements, ground truth images and measurement masks. [2]

## Installation
1. Clone the repository:
   git clone git@github.com:annikniem/NPGD.git

## Results
These are the accelerated MRI reconstruction results using the code in this repository. The results are presented along with corresponding ground truth images and simple k-space reconstructions.

![NPGD_examples](https://github.com/user-attachments/assets/3c9ca5aa-daef-42fb-9261-37d6e473c9b7)


## References
[1] Mardani, M., Sun, Q., Vasanawala, S. et al., 2018, Neural Proximal Gradient Descent for Compressive Imaging, https://arxiv.org/pdf/1806.03963

[2] Facebook and NYU FastMRI knee dataset, 2020, https://github.com/facebookresearch/fastMRI
