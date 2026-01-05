"""
Visualization functions for plotting training curves, confusion matrices, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RESULTS_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                         save_path=None, title="Training Curves"):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the figure
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, 
                         title="Confusion Matrix"):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_filters(filters, save_path=None, title="Convolutional Filters"):
    """
    Plot convolutional filters.
    
    Args:
        filters: Filter weights (num_filters, channels, height, width)
        save_path: Path to save the figure
        title: Plot title
    """
    num_filters = filters.shape[0]
    n_cols = 8
    n_rows = (num_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2 * n_rows))
    axes = axes.flatten() if num_filters > 1 else [axes]
    
    for i in range(num_filters):
        filter_img = filters[i]
        if filter_img.shape[0] == 3:  # RGB
            filter_img = np.transpose(filter_img, (1, 2, 0))
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
        else:  # Grayscale
            filter_img = filter_img[0]
        
        axes[i].imshow(filter_img, cmap='gray' if len(filter_img.shape) == 2 else None)
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i+1}')
    
    # Hide extra subplots
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_activation_maps(activations, save_path=None, title="Activation Maps"):
    """
    Plot activation maps from convolutional layers.
    
    Args:
        activations: Activation maps (batch, channels, height, width)
        save_path: Path to save the figure
        title: Plot title
    """
    # Take first sample from batch
    activations = activations[0]
    num_channels = activations.shape[0]
    
    n_cols = 8
    n_rows = (num_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2 * n_rows))
    axes = axes.flatten() if num_channels > 1 else [axes]
    
    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(activations[i], cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Channel {i+1}')
    
    # Hide extra subplots
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()
