"""
Data augmentation pipeline for deep learning.
"""

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import DATA_DIR


def get_augmentation_transforms(augment=True):
    """
    Get data augmentation transforms.
    
    Args:
        augment: Whether to apply augmentation
    
    Returns:
        transform: Transform pipeline
    """
    if augment:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.5)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    return transform


def visualize_augmentations(dataset, num_samples=8):
    """
    Visualize data augmentations.
    
    Args:
        dataset: Dataset to visualize
        num_samples: Number of samples to show
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))
    
    # Original transforms (no augmentation)
    original_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Augmented transforms
    aug_transform = get_augmentation_transforms(augment=True)
    
    for i in range(num_samples):
        # Get original image
        img, label = dataset[i]
        img_original = original_transform(img)
        
        # Get augmented image
        img_aug = aug_transform(img)
        
        # Denormalize for visualization
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        
        img_original = img_original * std + mean
        img_original = torch.clamp(img_original, 0, 1)
        img_original = img_original.permute(1, 2, 0).numpy()
        
        img_aug = img_aug * std + mean
        img_aug = torch.clamp(img_aug, 0, 1)
        img_aug = img_aug.permute(1, 2, 0).numpy()
        
        # Plot
        axes[0, i].imshow(img_original)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(img_aug)
        axes[1, i].set_title('Augmented')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'results', 'section4', 'augmentation_examples.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved augmentation visualization to {save_path}")
    plt.close()


if __name__ == '__main__':
    print("Loading CIFAR-10 dataset for augmentation visualization...")
    dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, 
                              transform=transforms.ToTensor())
    
    print("Visualizing augmentations...")
    visualize_augmentations(dataset, num_samples=8)
    print("Done!")
