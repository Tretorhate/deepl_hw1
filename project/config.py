"""
Centralized configuration and hyperparameters for the Deep Learning project.
"""

import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Section 1: MLP Configuration
SECTION1_CONFIG = {
    'hidden_sizes': [128, 64],
    'activation': 'relu',
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 20,
    'optimizer': 'adam',
    'loss': 'cross_entropy',
}

# Section 2: Optimization Configuration
SECTION2_CONFIG = {
    'learning_rates': [0.001, 0.01, 0.1],
    'optimizers': ['sgd', 'adam', 'rmsprop'],
    'epochs': 20,
    'batch_size': 64,
    'weight_decay': 0.0001,  # L2 regularization
}

# Section 3: CNN Configuration
SECTION3_CONFIG = {
    'num_filters': [32, 64, 128],
    'kernel_sizes': [3, 5],
    'pool_sizes': [2],
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
}

# Section 4: ResNet Configuration
SECTION4_CONFIG = {
    'num_layers': [18, 34, 50],
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'pretrained': True,
}

# Device configuration
# Note: We check torch.cuda.is_available() at runtime in the actual code
# This is just a default - actual device selection happens in section4/resnet.py
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'  # Fallback if torch not installed

# Random seed for reproducibility
RANDOM_SEED = 42

# Execution mode options:
# - QUICK_MODE = True: Fast testing (30-60 min) - skips some experiments
# - HYBRID_MODE = True: Balanced mode (2-4 hours) - all experiments, reduced epochs
# - Both False: Full mode (6-11 hours) - complete experiments with full epochs
QUICK_MODE = False  # Set to True for quick testing only
HYBRID_MODE = True  # Set to True for balanced mode (recommended)

# Hybrid mode configurations (used when HYBRID_MODE = True)
# Runs all required experiments but with reduced epochs/datasets for faster execution
if HYBRID_MODE:
    SECTION1_CONFIG = {
        'hidden_sizes': [128, 64],
        'activation': 'relu',
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 5,  # Reduced from 10 for faster execution (minimal quality mode)
        'optimizer': 'adam',
        'loss': 'cross_entropy',
    }
    
    SECTION2_CONFIG = {
        'learning_rates': [0.001, 0.01, 0.1],
        'optimizers': ['sgd', 'adam', 'rmsprop'],  # All 3 optimizers
        'epochs': 5,  # Reduced from 10 for faster execution (minimal quality mode)
        'batch_size': 64,
        'weight_decay': 0.0001,
    }
    
    SECTION3_CONFIG = {
        'num_filters': [32, 64, 128],
        'kernel_sizes': [3, 5],
        'pool_sizes': [2],
        'epochs': 3,  # Reduced from 5 for faster execution (minimal quality mode)
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
    }
    
    SECTION4_CONFIG = {
        'num_layers': [18, 34, 50],
        'epochs': 3,  # Reduced from 5 for faster execution (minimal quality mode)
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'pretrained': True,
    }

# Quick mode configurations (used when QUICK_MODE = True)
# Only for testing - skips some experiments
elif QUICK_MODE:
    SECTION1_CONFIG = {
        'hidden_sizes': [64, 32],  # Smaller network
        'activation': 'relu',
        'learning_rate': 0.001,
        'batch_size': 128,  # Larger batch for faster training
        'epochs': 3,  # Reduced from 20
        'optimizer': 'adam',
        'loss': 'cross_entropy',
    }
    
    SECTION2_CONFIG = {
        'learning_rates': [0.001, 0.01, 0.1],
        'optimizers': ['sgd', 'adam', 'rmsprop'],
        'epochs': 3,  # Reduced from 20
        'batch_size': 128,
        'weight_decay': 0.0001,
    }
    
    SECTION3_CONFIG = {
        'num_filters': [32, 64, 128],
        'kernel_sizes': [3, 5],
        'pool_sizes': [2],
        'epochs': 3,  # Reduced from 10
        'batch_size': 64,  # Larger batch
        'learning_rate': 0.001,
        'optimizer': 'adam',
    }
    
    SECTION4_CONFIG = {
        'num_layers': [18, 34, 50],
        'epochs': 3,  # Reduced from 10
        'batch_size': 64,  # Larger batch
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'pretrained': True,
    }
