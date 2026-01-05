# Deep Learning Course Project
**Authors:** Arsen Zharylkasynov, Sagi Yerassyl  
**Group:** IT-2301

A comprehensive deep learning project implementing neural networks from scratch using NumPy and modern architectures using PyTorch.

## Project Structure

```
project/
├── data/                    # Datasets (MNIST, Fashion-MNIST, CIFAR-10)
├── results/                 # Experiment results and plots
│   ├── section1/
│   ├── section2/
│   ├── section3/
│   └── section4/
├── section1/                # Foundations: MLP, Activations (20 pts)
├── section2/                # Optimization: Optimizers, Backprop (25 pts)
├── section3/                # CNNs: Conv2D, Pooling from scratch (30 pts)
├── section4/                # Modern: ResNet, Transfer Learning (25 pts)
├── utils/                   # Data loaders and visualization
├── config.py                # Configuration and execution modes
├── main.py                  # Main script
└── requirements.txt         # Dependencies
```

## Setup

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Configure execution mode** in `config.py`:
   - `QUICK_MODE = True`: Fast testing (30-60 min) - some experiments skipped
   - `HYBRID_MODE = True`: Balanced mode (1-2 hours) - all experiments, reduced epochs
   - Both `False`: Full mode (6-11 hours) - complete experiments

## Running Experiments

### Run all sections:

```bash
python main.py --all
```

### Run individual sections:

```bash
python main.py 1    # Section 1: Foundations
python main.py 2    # Section 2: Optimization
python main.py 3    # Section 3: CNNs
python main.py 4    # Section 4: Modern Architectures
```

### Interactive menu:

```bash
python main.py      # Shows menu to select sections
```

## Section Details

### Section 1: Foundations (20 points)

- **MLP Implementation**: Complete MLP from scratch with backpropagation
- **Activation Functions**: Sigmoid, Tanh, ReLU, LeakyReLU (all from scratch)
- **Universal Approximation**: Demonstrates network capacity to approximate non-linear functions
- **XOR Problem**: Solved with 2-layer network
- **Results**: Activation comparison plots, universal approximation visualizations

### Section 2: Optimization (25 points)

- **Backpropagation Derivation**: Complete mathematical derivation (LaTeX document)
- **Gradient Checking**: Numerical verification of backpropagation
- **Optimizers**: SGD, Adam, RMSprop implemented from scratch
- **Learning Rate Schedules**: Constant, Step Decay, Exponential, Polynomial
- **Gradient Flow Analysis**: Vanishing gradient analysis across network depths
- **Results**: Optimizer comparisons, LR schedule analysis, gradient flow plots

### Section 3: CNNs (30 points)

- **CNN from Scratch**: Conv2D, MaxPool, AvgPool implemented using NumPy
- **Full Training**: Complete CNN training on MNIST (target: 95% accuracy)
- **Filter Visualization**: Learned filter visualization at different layers
- **Receptive Field Analysis**: Theoretical receptive field calculations
- **Pooling Comparison**: MaxPool vs Average Pooling performance analysis
- **Results**: Training curves, filter visualizations, receptive field analysis, pooling comparison

### Section 4: Modern Architectures (25 points)

- **ResNet Implementation**: ResNet-18 with skip connections (PyTorch)
- **ResNet vs Plain Network**: Comparison demonstrating skip connection benefits
- **Transfer Learning**: Fine-tuning pretrained ResNet-18 on CIFAR-10
- **Data Augmentation**: Comprehensive augmentation pipeline
- **Results**: ResNet training curves, ResNet vs Plain comparison, transfer learning results

## Execution Modes

### Quick Mode (`QUICK_MODE = True`)

- **Time**: 30-60 minutes
- **Use**: Fast testing, code verification
- **Features**: Reduced epochs, smaller datasets, some experiments skipped
- **Accuracy**: Lower (demonstrates functionality)

### Hybrid Mode (`HYBRID_MODE = True`) - **Recommended**

- **Time**: 1-2 hours
- **Use**: Project submission, balanced performance
- **Features**: All experiments run, optimized epochs/datasets
- **Accuracy**: Good (meets all requirements)

### Full Mode (`QUICK_MODE = False, HYBRID_MODE = False`)

- **Time**: 6-11 hours
- **Use**: Best results, maximum accuracy
- **Features**: Full datasets, full epochs, all experiments
- **Accuracy**: Best (95%+ for CNN)

## Outputs

All results are saved to `results/` directory:

- Training/validation curves
- Confusion matrices
- Filter visualizations
- Activation maps
- Architecture comparisons
- Analysis plots

## Key Features

- **From Scratch**: Sections 1-3 use only NumPy (no high-level APIs)
- **Complete Backpropagation**: Full gradient computation for all layers
- **Mathematical Derivation**: Complete backpropagation derivation in LaTeX
- **Comprehensive Visualizations**: All experiments generate plots
- **GPU Support**: Automatic GPU detection for Section 4 (PyTorch)
- **Reproducible**: Fixed random seeds for all experiments

## Requirements Compliance

**Section 1 (20 pts)**: MLP, 4 activations, universal approximation  
**Section 2 (25 pts)**: Backprop derivation, gradient checking, 3 optimizers  
**Section 3 (30 pts)**: CNN from scratch, filter viz, receptive field, pooling comparison  
**Section 4 (25 pts)**: ResNet, ResNet vs Plain, transfer learning, augmentation

## Notes

- Datasets are downloaded automatically on first run
- All code includes documentation and type hints
- Results may vary slightly between runs due to randomness
- For best CNN accuracy (95%+), use Full Mode with full dataset
- Hybrid Mode is recommended for project submission (meets all requirements)

## Compiling Report

The LaTeX report (`report.tex`) can be compiled to PDF using:

```bash
pdflatex report.tex
pdflatex report.tex  # Run twice for references
```

Or use online tools like Overleaf.
