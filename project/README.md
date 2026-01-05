# Deep Learning Project
A comprehensive deep learning project implementing neural networks from scratch and using modern frameworks.

## Project Structure

```
project/
├── data/                    # Datasets (MNIST, Fashion-MNIST, CIFAR-10)
├── results/                 # Experiment results and plots
│   ├── section1/
│   ├── section2/
│   ├── section3/
│   └── section4/
├── models/                  # Saved model weights
├── utils/                   # Utility functions
│   ├── data_loader.py      # Dataset loaders
│   └── visualization.py    # Plotting functions
├── section1/                # Section 1: Foundations (20 points)
│   ├── activations.py      # Activation functions from scratch
│   ├── mlp.py              # MLP implementation
│   └── train_mlp.py        # Training scripts
├── section2/                # Section 2: Optimization (25 points)
│   ├── optimizers.py       # SGD, Adam, RMSprop from scratch
│   └── gradient_check.py   # Numerical gradient verification
├── section3/                # Section 3: CNNs (30 points)
│   └── cnn_layers.py       # Conv2D, MaxPool, AvgPool from scratch
├── section4/                # Section 4: Modern Architectures (25 points)
│   ├── resnet.py           # ResNet implementation
│   └── augmentation.py     # Data augmentation pipeline
├── config.py                # Centralized configuration
├── main.py                  # Main script to run experiments
└── requirements.txt         # Dependencies
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. The project uses the virtual environment from the parent folder (if available).

## Running Experiments

### Run all sections:

```bash
python main.py --all
```

### Run individual sections:

```bash
python main.py --section1    # MLP and activations
python main.py --section2    # Optimization
python main.py --section3    # CNNs
python main.py --section4    # Modern architectures
```

### Run individual scripts:

```bash
# Section 1
python section1/train_mlp.py

# Section 2
python section2/optimizers.py
python section2/gradient_check.py

# Section 3
python section3/cnn_layers.py

# Section 4
python section4/resnet.py
```

## Section Details

### Section 1: Foundations (20 points)

- Implement activation functions (Sigmoid, Tanh, ReLU, LeakyReLU) from scratch
- Build complete MLP with backpropagation
- Train on MNIST and XOR problem
- Compare different activation functions

### Section 2: Optimization (25 points)

- Implement optimizers (SGD, Adam, RMSprop) from scratch
- Numerical gradient verification
- Compare optimizer performance
- Analyze gradient flow and vanishing gradients
- Regularization techniques

### Section 3: CNNs (30 points)

- Implement Conv2D, MaxPool, AvgPool from scratch using NumPy
- Train CNN on CIFAR-10
- Visualize filters and activation maps
- Analyze receptive fields
- Compare different architectures

### Section 4: Modern Architectures (25 points)

- Implement ResNet with skip connections using PyTorch
- Data augmentation pipeline
- Transfer learning
- Fine-tuning pre-trained models

## Outputs

All results are saved to the `results/` directory:

- Training curves and loss plots
- Confusion matrices
- Filter visualizations
- Activation maps
- Architecture comparisons

## Notes

- Uses PyTorch for Section 4 (modern architectures)
- Sections 1-3 use NumPy implementations from scratch
- GPU is used automatically if available
- Datasets are downloaded automatically on first run
