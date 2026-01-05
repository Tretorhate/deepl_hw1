# Hybrid Mode Guide

## Overview

**Hybrid Mode** is the recommended mode for project submission. It runs **ALL required experiments** to meet project requirements, but with optimized epochs and dataset sizes for faster execution (2-4 hours instead of 6-11 hours).

## How to Enable

Edit `project/config.py` and set:
```python
QUICK_MODE = False
HYBRID_MODE = True  # Recommended for submission
```

## What Hybrid Mode Does

### All Required Experiments Run

**Section 1: Foundations**
- **All 4 activation functions** compared (sigmoid, tanh, ReLU, LeakyReLU)
- **Universal approximation study** with 3 network sizes
- **XOR problem** solved
- **Optimized**: 10 epochs (instead of 20), 25k samples (instead of 50k)

**Section 2: Optimization**
- **All 3 optimizers** compared (SGD, Adam, RMSprop)
- **All 4 learning rate schedules** tested
- **Gradient flow analysis** with 3 network depths
- **Optimized**: 10 epochs (instead of 20), 25k samples (instead of 50k)

**Section 3: CNNs**
- **Complete CNN training** (not skipped!)
- **Filter visualization**
- **Receptive field analysis**
- **Pooling strategy comparison** (MaxPool vs AvgPool)
- **Optimized**: 5 epochs (instead of 10), 15k samples (instead of full MNIST)

**Section 4: Modern Architectures**
- **ResNet training**
- **Plain Network comparison** (not skipped!)
- **Transfer Learning** with pretrained model (not skipped!)
- **Data augmentation pipeline**
- **Optimized**: 5 epochs (instead of 10), 15k training samples

## Execution Time Estimates

### Hybrid Mode (HYBRID_MODE = True)
- **Section 1**: ~45-60 minutes
- **Section 2**: ~30-45 minutes
- **Section 3**: ~45-60 minutes
- **Section 4**: ~30-45 minutes
- **Total**: ~2.5-4 hours

### Comparison with Other Modes

| Mode | Time | Requirements Met | Use Case |
|------|------|------------------|----------|
| **Quick Mode** | 30-60 min | Some skipped | Testing only |
| **Hybrid Mode** | 2-4 hours | **All met** | **Submission** |
| **Full Mode** | 6-11 hours | All met | Best results |

## Requirements Compliance

### Section 1 (20 points)
- MLP Implementation (12 pts) - Complete
- Activation Function Comparison (5 pts) - **All 4 activations**
- Universal Approximation Study (3 pts) - Complete

### Section 2 (25 points)
- Backpropagation Derivation (10 pts) - Document included
- Gradient Checking (5 pts) - Complete
- Optimization Algorithms Comparison (10 pts) - **All 3 optimizers**

### Section 3 (30 points)
- CNN Implementation (15 pts) - **Full training runs**
- Filter Visualization (8 pts) - Complete
- Receptive Field Study (4 pts) - Complete
- Pooling Strategies Comparison (3 pts) - Complete

### Section 4 (25 points)
- ResNet Implementation (10 pts) - **With Plain Network comparison**
- Transfer Learning (10 pts) - **Complete implementation**
- Data Augmentation (5 pts) - Complete

## Running in Hybrid Mode

```bash
# Run all sections
python main.py --all

# Or run individually
python main.py 1
python main.py 2
python main.py 3
python main.py 4
```

## Expected Results

### Section 1
- Activation comparison plots with all 4 functions
- Universal approximation visualizations
- XOR problem solved

### Section 2
- Optimizer comparison with all 3 optimizers
- Learning rate schedule analysis
- Gradient flow analysis

### Section 3
- CNN training curves (may not reach 95% with 5 epochs, but demonstrates functionality)
- Filter visualizations
- Receptive field analysis
- Pooling comparison

### Section 4
- ResNet vs Plain Network comparison
- Transfer learning results
- All visualizations

## Tips

1. **Run overnight**: If you have time, hybrid mode can run overnight
2. **Monitor progress**: Check console output to see progress
3. **Check results**: All results saved to `results/` directory
4. **For best accuracy**: Use Full Mode if you need 95% CNN accuracy

## Notes

- Hybrid mode meets **all project requirements**
- All experiments run (nothing skipped)
- Results may be slightly lower than full mode (expected with fewer epochs)
- Perfect balance between completeness and execution time
- **Recommended for project submission**

## When to Use Full Mode

Use Full Mode (`QUICK_MODE = False, HYBRID_MODE = False`) only if:
- You need maximum accuracy (e.g., 95% for CNN)
- You have 6-11 hours available
- You want the best possible results

For most cases, **Hybrid Mode is sufficient and recommended**.
