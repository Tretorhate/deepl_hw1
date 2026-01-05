"""
Activation functions implemented from scratch: Sigmoid, Tanh, ReLU, LeakyReLU
"""

import numpy as np


class Activation:
    """Base class for activation functions."""
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError


class Sigmoid(Activation):
    """Sigmoid activation function: σ(x) = 1 / (1 + exp(-x))"""
    
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)


class Tanh(Activation):
    """Hyperbolic tangent activation function: tanh(x)"""
    
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)


class ReLU(Activation):
    """Rectified Linear Unit: ReLU(x) = max(0, x)"""
    
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, grad_output):
        return grad_output * (self.output > 0).astype(float)


class LeakyReLU(Activation):
    """Leaky ReLU: LeakyReLU(x) = max(αx, x) where α is typically 0.01"""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        self.input = x
        self.output = np.maximum(self.alpha * x, x)
        return self.output
    
    def backward(self, grad_output):
        return grad_output * np.where(self.input > 0, 1, self.alpha)


# Convenience functions for direct use
def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    """Tanh function."""
    return np.tanh(x)


def relu(x):
    """ReLU function."""
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU function."""
    return np.maximum(alpha * x, x)


if __name__ == '__main__':
    # Test activation functions
    x = np.linspace(-5, 5, 100)
    
    activations = {
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh(),
        'ReLU': ReLU(),
        'LeakyReLU': LeakyReLU(alpha=0.01)
    }
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, activation) in enumerate(activations.items()):
        y = activation.forward(x)
        axes[idx].plot(x, y, linewidth=2)
        axes[idx].set_title(f'{name} Activation')
        axes[idx].grid(True)
        axes[idx].axhline(0, color='black', linestyle='--', linewidth=0.5)
        axes[idx].axvline(0, color='black', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    print("Activation functions plotted and saved to activation_functions.png")
