import sys

print("=" * 60)
print("Detailed Installation Verification")
print("=" * 60)

# Check PyTorch
try:
    import torch
    print(f"\n✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version (PyTorch): {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"\n✗ PyTorch error: {e}")
    sys.exit(1)

# Check MMEngine
try:
    import mmengine
    print(f"\n✓ MMEngine: {mmengine.__version__}")
except Exception as e:
    print(f"\n✗ MMEngine error: {e}")
    sys.exit(1)

# Check MMCV
try:
    import mmcv
    print(f"\n✓ MMCV: {mmcv.__version__}")
    
    # Test MMCV ops
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
    print(f"  MMCV CUDA version: {get_compiling_cuda_version()}")
    print(f"  MMCV compiler: {get_compiler_version()}")
    
    # Try to import CUDA ops
    try:
        from mmcv.ops import RoIAlign
        print(f"  ✓ MMCV CUDA ops available")
    except Exception as e:
        print(f"  ✗ MMCV CUDA ops error: {e}")
        
except Exception as e:
    print(f"\n✗ MMCV error: {e}")
    print("\nThis is the DLL error. MMCV and PyTorch versions don't match.")
    print("Try reinstalling with matching versions.")
    sys.exit(1)

# Check MMDetection
try:
    import mmdet
    print(f"\n✓ MMDetection: {mmdet.__version__}")
    
    # Try to import key components
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS
    print(f"  ✓ MMDetection imports working")
    
except Exception as e:
    print(f"\n✗ MMDetection error: {e}")
    sys.exit(1)

# Check other dependencies
try:
    import cv2
    print(f"\n✓ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"\n✗ OpenCV error: {e}")

try:
    from pycocotools import coco
    print(f"✓ pycocotools available")
except Exception as e:
    print(f"✗ pycocotools error: {e}")

print("\n" + "=" * 60)
print("All checks passed! Ready to train.")
print("=" * 60)