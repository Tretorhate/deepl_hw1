"""
CUDA/GPU Diagnostic Script
Tests GPU availability and provides detailed diagnostics for troubleshooting.
"""

import sys
import os

print("=" * 70)
print("CUDA/GPU Diagnostic Test")
print("=" * 70)
print()

# Test 1: Check if PyTorch is installed
print("1. Checking PyTorch Installation...")
print("-" * 70)
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    pytorch_installed = True
except ImportError:
    print("   ERROR: PyTorch is not installed!")
    print("   Install with: pip install torch torchvision")
    pytorch_installed = False
    sys.exit(1)

print()

# Test 2: Check CUDA availability
print("2. Checking CUDA Availability...")
print("-" * 70)
if pytorch_installed:
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"   CUDA version (PyTorch): {torch.version.cuda}")
        if hasattr(torch.backends.cudnn, 'version'):
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"     Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"     Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"     Compute Capability: {props.major}.{props.minor}")
            print(f"     Multiprocessors: {props.multi_processor_count}")
    else:
        print("   CUDA is NOT available")
        print()
        print("   Possible reasons:")
        print("   - PyTorch was installed without CUDA support (CPU-only)")
        print("   - CUDA drivers are not installed")
        print("   - CUDA version mismatch")
        print("   - GPU not detected by system")

print()

# Test 3: Check NVIDIA drivers
print("3. Checking NVIDIA Drivers...")
print("-" * 70)
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   NVIDIA drivers are installed")
        # Extract GPU name from nvidia-smi output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'NVIDIA' in line and 'Driver' in line:
                print(f"   {line.strip()}")
            if 'GeForce' in line or 'Quadro' in line or 'Tesla' in line or 'RTX' in line:
                if '|' in line:
                    print(f"   GPU detected: {line.strip()}")
    else:
        print("   WARNING: nvidia-smi command failed")
        print("   NVIDIA drivers may not be installed")
except FileNotFoundError:
    print("   WARNING: nvidia-smi not found")
    print("   NVIDIA drivers may not be installed")
    print("   Install from: https://www.nvidia.com/drivers")
except subprocess.TimeoutExpired:
    print("   WARNING: nvidia-smi timed out")
except Exception as e:
    print(f"   Could not check NVIDIA drivers: {e}")

print()

# Test 4: Check environment variables
print("4. Checking Environment Variables...")
print("-" * 70)
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_visible:
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print("   NOTE: This restricts which GPUs PyTorch can see")
else:
    print("   CUDA_VISIBLE_DEVICES: Not set (all GPUs visible)")

cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
if cuda_home:
    print(f"   CUDA_HOME/CUDA_PATH: {cuda_home}")
else:
    print("   CUDA_HOME/CUDA_PATH: Not set")

print()

# Test 5: Test actual GPU computation
print("5. Testing GPU Computation...")
print("-" * 70)
if pytorch_installed and torch.cuda.is_available():
    try:
        # Create a tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   SUCCESS: GPU computation test passed!")
        print(f"   Test tensor shape: {z.shape}")
        print(f"   Test tensor device: {z.device}")
        print(f"   Test tensor dtype: {z.dtype}")
        
        # Test memory allocation
        print(f"\n   GPU Memory:")
        print(f"     Allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
        print(f"     Reserved: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")
        print(f"     Max allocated: {torch.cuda.max_memory_allocated(0) / 1e6:.2f} MB")
        
        gpu_test_passed = True
    except Exception as e:
        print(f"   ERROR: GPU computation test failed!")
        print(f"   Error: {e}")
        gpu_test_passed = False
else:
    print("   SKIPPED: CUDA not available")
    gpu_test_passed = False

print()

# Test 6: Check PyTorch installation type
print("6. Checking PyTorch Installation Type...")
print("-" * 70)
if pytorch_installed:
    # Check if it's CPU-only or CUDA version
    try:
        # Try to import CUDA functions
        from torch.cuda import is_available
        if torch.cuda.is_available():
            print("   PyTorch has CUDA support and GPU is available")
        else:
            print("   PyTorch has CUDA support but GPU is not available")
            print("   This might be:")
            print("   - CPU-only PyTorch installation")
            print("   - CUDA drivers not installed")
            print("   - CUDA version mismatch")
    except ImportError:
        print("   PyTorch appears to be CPU-only version")
        print("   Install CUDA version from: https://pytorch.org/get-started/locally/")

print()

# Summary and recommendations
print("=" * 70)
print("SUMMARY")
print("=" * 70)

if pytorch_installed and torch.cuda.is_available() and gpu_test_passed:
    print("STATUS: GPU is available and working!")
    print(f"Device: cuda:0 ({torch.cuda.get_device_name(0)})")
    print("\nYour code will use GPU for Section 4 (ResNet).")
elif pytorch_installed and torch.cuda.is_available() and not gpu_test_passed:
    print("STATUS: GPU detected but computation test failed!")
    print("There may be a compatibility issue.")
elif pytorch_installed and not torch.cuda.is_available():
    print("STATUS: GPU is NOT available")
    print("\nRECOMMENDATIONS:")
    print("1. Check if you have an NVIDIA GPU:")
    print("   - Run: nvidia-smi")
    print("   - If it fails, you may not have an NVIDIA GPU")
    print()
    print("2. If you have an NVIDIA GPU, install CUDA-enabled PyTorch:")
    print("   - Visit: https://pytorch.org/get-started/locally/")
    print("   - Select your CUDA version")
    print("   - Example: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("3. Verify CUDA installation:")
    print("   - Check CUDA version: nvcc --version")
    print("   - Check PyTorch CUDA: python -c \"import torch; print(torch.version.cuda)\"")
    print()
    print("4. For CPU-only systems:")
    print("   - Code will automatically use CPU")
    print("   - Training will be slower but will work")
    print("   - Sections 1-3 use NumPy (CPU-only anyway)")
else:
    print("STATUS: PyTorch not installed properly")

print()
print("=" * 70)
print("Note: Sections 1-3 use NumPy (CPU-only)")
print("      Only Section 4 (ResNet) uses GPU")
print("=" * 70)

# Test device selection logic
print()
print("=" * 70)
print("Device Selection Test")
print("=" * 70)

# Test the current config.py logic
print("\nCurrent config.py logic:")
print(f"  DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'")
print(f"  Result: {'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'}")

# Test correct logic
print("\nCorrect logic:")
if pytorch_installed:
    correct_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'")
    print(f"  Result: {correct_device}")
    
    if correct_device == 'cuda':
        print("\n  RECOMMENDATION: Update config.py to use:")
        print("  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'")
    else:
        print("\n  Code will use CPU (which is fine for this project)")

print()
print("=" * 70)
