"""
Multi-Layer Perceptron (MLP) implementation from scratch with backpropagation.
"""

import numpy as np
from .activations import Sigmoid, Tanh, ReLU, LeakyReLU


class LinearLayer:
    """Linear (fully connected) layer."""
    
    def __init__(self, input_size, output_size):
        # Xavier/Glorot initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        return grad_input, grad_weights, grad_bias
    
    def update(self, grad_weights, grad_bias, learning_rate):
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias


class MLP:
    """Multi-Layer Perceptron with configurable architecture."""
    
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        """
        Initialize MLP.
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output
            activation: Activation function ('sigmoid', 'tanh', 'relu', 'leaky_relu')
        """
        self.layers = []
        self.activations = []
        
        # Create layers
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(LinearLayer(sizes[i], sizes[i+1]))
            
            # Add activation (except for output layer)
            if i < len(sizes) - 2:
                if activation == 'sigmoid':
                    self.activations.append(Sigmoid())
                elif activation == 'tanh':
                    self.activations.append(Tanh())
                elif activation == 'relu':
                    self.activations.append(ReLU())
                elif activation == 'leaky_relu':
                    self.activations.append(LeakyReLU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
            else:
                self.activations.append(None)  # No activation for output layer
    
    def forward(self, x):
        """Forward pass."""
        for layer, activation in zip(self.layers, self.activations):
            x = layer.forward(x)
            if activation is not None:
                x = activation.forward(x)
        return x
    
    def backward(self, grad_output):
        """Backward pass (backpropagation)."""
        grads = []
        
        # Backward through layers in reverse
        for i in range(len(self.layers) - 1, -1, -1):
            # Backward through activation
            if self.activations[i] is not None:
                grad_output = self.activations[i].backward(grad_output)
            
            # Backward through linear layer
            grad_input, grad_weights, grad_bias = self.layers[i].backward(grad_output)
            grads.append((grad_weights, grad_bias))
            grad_output = grad_input
        
        return list(reversed(grads))
    
    def update(self, grads, learning_rate):
        """Update weights using gradients."""
        for layer, (grad_weights, grad_bias) in zip(self.layers, grads):
            layer.update(grad_weights, grad_bias, learning_rate)
    
    def predict(self, x):
        """Make predictions."""
        return self.forward(x)
    
    def get_weights(self):
        """Get all weights."""
        return [layer.weights for layer in self.layers]
    
    def set_weights(self, weights):
        """Set all weights."""
        for layer, w in zip(self.layers, weights):
            layer.weights = w


def cross_entropy_loss(y_pred, y_true):
    """
    Cross-entropy loss for multi-class classification.
    
    Args:
        y_pred: Predicted probabilities (batch_size, num_classes)
        y_true: True labels (batch_size,)
    
    Returns:
        loss: Scalar loss value
        grad: Gradient w.r.t. y_pred
    """
    batch_size = y_pred.shape[0]
    
    # Softmax
    exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
    # One-hot encode y_true
    y_one_hot = np.zeros_like(probs)
    y_one_hot[np.arange(batch_size), y_true] = 1
    
    # Loss
    loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-15), axis=1))
    
    # Gradient
    grad = (probs - y_one_hot) / batch_size
    
    return loss, grad


def softmax(x):
    """Softmax function."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
