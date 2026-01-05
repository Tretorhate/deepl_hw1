# Quick Mode Guide
## Overview

Quick Mode is enabled by default in `config.py` to allow faster testing of the project (30-60 minutes instead of 6-11 hours).

## How to Enable/Disable

Edit `project/config.py` and change:

```python
QUICK_MODE = True   # For quick testing (30-60 min)
QUICK_MODE = False  # For full experiments (6-11 hours)
```

## What Quick Mode Does

### Section 1: Foundations

- **Reduced epochs**: 3 instead of 20
- **Smaller dataset**: 10,000 samples instead of 50,000
- **Fewer activations**: Tests only ReLU and Tanh (instead of 4)
- **Universal approximation**: 2 networks × 300 epochs (instead of 5 networks × 2000 epochs)
- **XOR**: 200 epochs (instead of 1000)

### Section 2: Optimization

- **Reduced epochs**: 3 instead of 20
- **Smaller dataset**: 10,000 samples instead of 50,000
- **Fewer optimizers**: Tests only Adam (instead of 3)
- **Fewer LR schedules**: Tests only Constant and Step Decay (instead of 4)
- **Fewer network depths**: Tests 2 depths (instead of 4)

### Section 3: CNNs

- **Reduced epochs**: 3 instead of 10
- **Smaller dataset**: 5,000 samples instead of full MNIST
- **Larger batch size**: 64 instead of 32
- **Skipped**: Full CNN training (pooling comparison demonstrates functionality)
- **Pooling comparison**: 2 epochs (instead of 5)

### Section 4: Modern Architectures

- **Reduced epochs**: 3 instead of 10
- **Smaller dataset**: 5,000 training samples, 1,000 test samples
- **Larger batch size**: 64 instead of 32
- **Skipped**: Plain Network comparison and Transfer Learning (ResNet training demonstrates functionality)

## Estimated Execution Times

### Quick Mode (QUICK_MODE = True)

- **Section 1**: ~15-20 minutes
- **Section 2**: ~10-15 minutes
- **Section 3**: ~10-15 minutes
- **Section 4**: ~5-10 minutes
- **Total**: ~40-60 minutes

### Full Mode (QUICK_MODE = False)

- **Section 1**: ~2-3 hours
- **Section 2**: ~1.5-2 hours
- **Section 3**: ~2-4 hours
- **Section 4**: ~1-2 hours
- **Total**: ~6-11 hours

## Running in Quick Mode

```bash
# Run all sections in quick mode
python main.py --all

# Or run individual sections
python main.py 1
python main.py 2
python main.py 3
python main.py 4
```

## What You Still Get in Quick Mode

- All core functionality demonstrated
- All visualizations generated
- All analysis plots created
- Understanding of all concepts
- Results saved to `results/` directory

## When to Use Full Mode

Use full mode when:

- You need final project submission results
- You want to achieve target accuracies (e.g., 95% for CNN)
- You have time to run overnight
- You want complete comparisons

## Tips

1. **Start with Quick Mode**: Verify everything works first
2. **Run overnight**: If you need full results, run with `QUICK_MODE = False` overnight
3. **Run sections separately**: You can run one section at a time
4. **Monitor progress**: Check the console output to see progress

## Notes

- Quick mode still generates all required visualizations
- All code paths are tested in quick mode
- Results may differ slightly from full mode (expected)
- You can always switch to full mode later for final results
