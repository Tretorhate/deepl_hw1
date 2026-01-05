"""
Numerical gradient verification for backpropagation.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from section1.mlp import MLP, cross_entropy_loss
from config import RANDOM_SEED

np.random.seed(RANDOM_SEED)


def numerical_gradient(f, x, h=1e-5):
    """
    Compute numerical gradient using finite differences.
    
    Args:
        f: Function that takes x and returns a scalar
        x: Input array
        h: Step size for finite differences
    
    Returns:
        grad: Numerical gradient
    """
    grad = np.zeros_like(x)
    flat_x = x.flatten()
    flat_grad = grad.flatten()
    
    for i in range(len(flat_x)):
        # Forward difference
        x_plus = flat_x.copy()
        x_plus[i] += h
        f_plus = f(x_plus.reshape(x.shape))
        
        # Backward difference
        x_minus = flat_x.copy()
        x_minus[i] -= h
        f_minus = f(x_minus.reshape(x.shape))
        
        # Central difference
        flat_grad[i] = (f_plus - f_minus) / (2 * h)
    
    return flat_grad.reshape(x.shape)


def gradient_check(model, X, y, epsilon=1e-7):
    """
    Check gradients using numerical differentiation.
    
    Args:
        model: MLP model
        X: Input data
        y: True labels
        epsilon: Tolerance for gradient check
    
    Returns:
        is_correct: Whether gradients are correct
        max_error: Maximum relative error
    """
    # Forward and backward pass
    y_pred = model.forward(X)
    loss, grad = cross_entropy_loss(y_pred, y)
    grads = model.backward(grad)
    
    # Check gradients for each layer
    all_correct = True
    max_error = 0
    
    for layer_idx, (layer, (grad_weights, grad_bias)) in enumerate(zip(model.layers, grads)):
        # Check weight gradients
        def loss_fn_weights(w):
            old_weights = layer.weights.copy()
            layer.weights = w
            y_pred = model.forward(X)
            loss, _ = cross_entropy_loss(y_pred, y)
            layer.weights = old_weights
            return loss
        
        num_grad_weights = numerical_gradient(loss_fn_weights, layer.weights)
        rel_error_weights = np.abs(grad_weights - num_grad_weights) / (np.abs(grad_weights) + np.abs(num_grad_weights) + epsilon)
        max_rel_error_weights = np.max(rel_error_weights)
        
        # Check bias gradients
        def loss_fn_bias(b):
            old_bias = layer.bias.copy()
            layer.bias = b
            y_pred = model.forward(X)
            loss, _ = cross_entropy_loss(y_pred, y)
            layer.bias = old_bias
            return loss
        
        num_grad_bias = numerical_gradient(loss_fn_bias, layer.bias)
        rel_error_bias = np.abs(grad_bias - num_grad_bias) / (np.abs(grad_bias) + np.abs(num_grad_bias) + epsilon)
        max_rel_error_bias = np.max(rel_error_bias)
        
        max_error = max(max_error, max_rel_error_weights, max_rel_error_bias)
        
        print(f"Layer {layer_idx + 1}:")
        print(f"  Weights - Max relative error: {max_rel_error_weights:.2e}")
        print(f"  Bias - Max relative error: {max_rel_error_bias:.2e}")
        
        if max_rel_error_weights > epsilon or max_rel_error_bias > epsilon:
            all_correct = False
    
    return all_correct, max_error


if __name__ == '__main__':
    print("=" * 50)
    print("Gradient Check")
    print("=" * 50)
    print()
    
    # Create a small model for testing
    model = MLP(input_size=10, hidden_sizes=[5, 3], output_size=2, activation='relu')
    
    # Create dummy data
    X = np.random.randn(5, 10)
    y = np.random.randint(0, 2, size=5)
    
    print("Checking gradients...")
    print()
    
    is_correct, max_error = gradient_check(model, X, y)
    
    print()
    print("=" * 50)
    if is_correct:
        print("Gradients are correct!")
    else:
        print("Gradient check failed!")
    print(f"Maximum relative error: {max_error:.2e}")
    print("=" * 50)
