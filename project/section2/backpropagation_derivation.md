# Backpropagation Mathematical Derivation

## Overview
This document provides a complete mathematical derivation of the backpropagation algorithm for a 2-layer neural network.

## Network Architecture

Consider a 2-layer neural network:
- **Input layer**: $x \in \mathbb{R}^{d}$ (input vector of dimension $d$)
- **Hidden layer**: $h \in \mathbb{R}^{m}$ (hidden layer with $m$ neurons)
- **Output layer**: $\hat{y} \in \mathbb{R}^{k}$ (output vector of dimension $k$)

### Forward Pass

**Layer 1 (Input to Hidden):**
$$z_1 = W_1 x + b_1$$
$$h = \sigma(z_1)$$

where:
- $W_1 \in \mathbb{R}^{m \times d}$ is the weight matrix
- $b_1 \in \mathbb{R}^{m}$ is the bias vector
- $\sigma$ is the activation function (e.g., ReLU, sigmoid, tanh)

**Layer 2 (Hidden to Output):**
$$z_2 = W_2 h + b_2$$
$$\hat{y} = \text{softmax}(z_2)$$

where:
- $W_2 \in \mathbb{R}^{k \times m}$ is the weight matrix
- $b_2 \in \mathbb{R}^{k}$ is the bias vector
- $\text{softmax}$ is the softmax activation function

### Loss Function

For multi-class classification, we use cross-entropy loss:
$$L = -\sum_{i=1}^{k} y_i \log(\hat{y}_i)$$

where $y$ is the one-hot encoded true label.

## Backpropagation Derivation

### Step 1: Gradient w.r.t. Output Layer

We start by computing the gradient of the loss with respect to the output logits $z_2$:

$$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2}$$

**Gradient of loss w.r.t. softmax output:**
$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}$$

**Gradient of softmax w.r.t. logits:**
$$\frac{\partial \hat{y}_i}{\partial z_{2,j}} = \begin{cases}
\hat{y}_i (1 - \hat{y}_i) & \text{if } i = j \\
-\hat{y}_i \hat{y}_j & \text{if } i \neq j
\end{cases}$$

Combining these:
$$\frac{\partial L}{\partial z_{2,i}} = \sum_{j=1}^{k} \frac{\partial L}{\partial \hat{y}_j} \cdot \frac{\partial \hat{y}_j}{\partial z_{2,i}}$$

$$= -\sum_{j=1}^{k} \frac{y_j}{\hat{y}_j} \cdot \frac{\partial \hat{y}_j}{\partial z_{2,i}}$$

$$= -\frac{y_i}{\hat{y}_i} \cdot \hat{y}_i (1 - \hat{y}_i) + \sum_{j \neq i} \frac{y_j}{\hat{y}_j} \cdot \hat{y}_i \hat{y}_j$$

$$= -y_i (1 - \hat{y}_i) + \sum_{j \neq i} y_j \hat{y}_i$$

$$= -y_i + y_i \hat{y}_i + \sum_{j \neq i} y_j \hat{y}_i$$

$$= -y_i + \hat{y}_i \sum_{j=1}^{k} y_j$$

Since $\sum_{j=1}^{k} y_j = 1$ (one-hot encoding):
$$\frac{\partial L}{\partial z_2} = \hat{y} - y$$

### Step 2: Gradient w.r.t. $W_2$ and $b_2$

**Gradient w.r.t. $W_2$:**
$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial W_2}$$

Since $z_2 = W_2 h + b_2$:
$$\frac{\partial z_2}{\partial W_2} = h^T$$

Therefore:
$$\frac{\partial L}{\partial W_2} = (\hat{y} - y) \cdot h^T$$

In matrix form:
$$\frac{\partial L}{\partial W_2} = (\hat{y} - y) h^T$$

**Gradient w.r.t. $b_2$:**
$$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial b_2} = \hat{y} - y$$

### Step 3: Gradient w.r.t. Hidden Layer

**Gradient w.r.t. $h$:**
$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial h}$$

Since $z_2 = W_2 h + b_2$:
$$\frac{\partial z_2}{\partial h} = W_2^T$$

Therefore:
$$\frac{\partial L}{\partial h} = W_2^T (\hat{y} - y)$$

### Step 4: Gradient w.r.t. $z_1$

**Gradient w.r.t. $z_1$:**
$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial z_1}$$

Since $h = \sigma(z_1)$:
$$\frac{\partial h}{\partial z_1} = \sigma'(z_1)$$

where $\sigma'$ is the derivative of the activation function.

Therefore:
$$\frac{\partial L}{\partial z_1} = W_2^T (\hat{y} - y) \odot \sigma'(z_1)$$

where $\odot$ denotes element-wise multiplication.

### Step 5: Gradient w.r.t. $W_1$ and $b_1$

**Gradient w.r.t. $W_1$:**
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$$

Since $z_1 = W_1 x + b_1$:
$$\frac{\partial z_1}{\partial W_1} = x^T$$

Therefore:
$$\frac{\partial L}{\partial W_1} = [W_2^T (\hat{y} - y) \odot \sigma'(z_1)] x^T$$

In matrix form:
$$\frac{\partial L}{\partial W_1} = [W_2^T (\hat{y} - y) \odot \sigma'(z_1)] x^T$$

**Gradient w.r.t. $b_1$:**
$$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1} = W_2^T (\hat{y} - y) \odot \sigma'(z_1)$$

## Summary of Gradients

For a 2-layer network with cross-entropy loss and softmax output:

1. **Output layer gradients:**
   - $\frac{\partial L}{\partial z_2} = \hat{y} - y$
   - $\frac{\partial L}{\partial W_2} = (\hat{y} - y) h^T$
   - $\frac{\partial L}{\partial b_2} = \hat{y} - y$

2. **Hidden layer gradients:**
   - $\frac{\partial L}{\partial h} = W_2^T (\hat{y} - y)$
   - $\frac{\partial L}{\partial z_1} = W_2^T (\hat{y} - y) \odot \sigma'(z_1)$
   - $\frac{\partial L}{\partial W_1} = [W_2^T (\hat{y} - y) \odot \sigma'(z_1)] x^T$
   - $\frac{\partial L}{\partial b_1} = W_2^T (\hat{y} - y) \odot \sigma'(z_1)$

## Activation Function Derivatives

**ReLU:**
$$\sigma(x) = \max(0, x)$$
$$\sigma'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Sigmoid:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Tanh:**
$$\sigma(x) = \tanh(x)$$
$$\sigma'(x) = 1 - \tanh^2(x) = 1 - \sigma^2(x)$$

## Implementation Notes

1. The gradients are computed in reverse order (output to input)
2. The chain rule is applied at each step
3. Element-wise operations (like $\odot$) are crucial for correct computation
4. For mini-batch training, gradients are averaged over the batch

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
