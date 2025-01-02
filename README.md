# CIFAR-10 Image Classification Using CNN and MLP

This project involves implementing and comparing two neural network architectures—a Convolutional Neural Network (CNN) and a Multi-Layer Perceptron (MLP)—for image classification on a subset of the CIFAR-10 dataset using PyTorch.

## Overview:

The goal of this project is to classify images from three selected classes of the CIFAR-10 dataset using two different neural network models—a CNN and an MLP—and to compare their performances. The project demonstrates the effectiveness of CNNs over MLPs in handling image data due to the ability of CNNs to capture spatial hierarchies.

## Dataset Preparation:

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes. For this project:

Selected Classes: Airplane, Automobile, and Bird.
Total Images Used: Approximately 18,000 images (6,000 per class).
Dataset Splits:
Training Set: 12,000 images (4,000 per class) – 80% of the 15,000 training images.
Validation Set: 3,000 images (1,000 per class) – 20% of the 15,000 training images.
Test Set: 3,000 images (1,000 per class) retained from the original CIFAR-10 test set.
Methodology:

## Data Preparation:

Data Loading: Used PyTorch's torchvision.datasets.CIFAR10 to load the dataset and extracted images corresponding to the selected classes.
Custom Dataset Class: Created a custom Dataset class to handle data transformations and facilitate integration with PyTorch data loaders.
Stratified Split: Performed a stratified random split to maintain equal class representation across training, validation, and test sets.
Data Loaders: Created data loaders for training, validation, and test sets with appropriate batch sizes for efficient mini-batch processing.
## CNN Architecture:

The CNN was designed to capture spatial features effectively:

First Convolutional Layer:
Kernel Size: 5x5
Output Channels: 16
Stride: 1
Padding: 1 (to preserve spatial dimensions)
Activation: ReLU
Max-Pooling Layer: Kernel Size 3x3, Stride 2
Second Convolutional Layer:
Kernel Size: 3x3
Output Channels: 32
Stride: 1
Padding: 0
Activation: ReLU
Max-Pooling Layer: Kernel Size 3x3, Stride 3
Fully Connected Layers:
Flattened output from convolutional layers.
First Fully Connected Layer: 16 neurons with ReLU activation.
Output Layer (Classification Head): Neurons equal to the number of classes (3).
## MLP Architecture:

The MLP model served as a baseline for comparison:

Input Layer:
Images flattened into one-dimensional tensors.
First Fully Connected Layer:
64 neurons.
Activation: ReLU.
Output Layer (Classification Head):
Neurons equal to the number of classes (3).
## Training Process:

Loss Function: Cross-Entropy Loss, suitable for multi-class classification.
Optimizer: Adam optimizer, which adapts learning rates during training.
Training Epochs: Both models were trained for 15 epochs.
Logging: Training and validation losses and accuracies were recorded after each epoch.
Model Checkpoints: Saved models as .pth files for reproducibility.
## Evaluation:

Performance Metrics:
Accuracy: Measures overall correctness.
F1-Score: Harmonic mean of precision and recall.
Confusion Matrix: Plotted for training, validation, and test sets to visualize prediction distributions.
Analysis: Compared both models based on metrics and confusion matrices.


## Prerequisites:

Python 3.x
PyTorch
torchvision
NumPy
Matplotlib
scikit-learn

Link for dataset:https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
