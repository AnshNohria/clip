#!/usr/bin/env python3
"""
Package Verification Script

Verifies that all required packages for the RemoteCLIP pipeline are installed
and can be imported correctly.
"""
import sys

def check_import(module_name, package_name=None, submodule=None):
    """Check if a module can be imported."""
    display_name = package_name or module_name
    try:
        if submodule:
            module = __import__(module_name, fromlist=[submodule])
            getattr(module, submodule)
        else:
            __import__(module_name)
        print(f"  ✓ {display_name}")
        return True
    except ImportError as e:
        print(f"  ✗ {display_name} - {e}")
        return False
    except Exception as e:
        print(f"  ⚠ {display_name} - Imported but error: {e}")
        return True


def main():
    print("=" * 60)
    print("REMOTECLIP PIPELINE - PACKAGE VERIFICATION")
    print("=" * 60)
    
    results = {"passed": 0, "failed": 0}
    
    # Core packages
    print("\n[1/8] Core PyTorch:")
    packages = [
        ("torch", None, None),
        ("torchvision", None, None),
    ]
    for mod, pkg, sub in packages:
        if check_import(mod, pkg, sub):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available - will use CPU")
    except:
        pass
    
    # Diffusion models
    print("\n[2/8] Diffusion Models:")
    packages = [
        ("diffusers", None, None),
        ("accelerate", None, None),
        ("safetensors", None, None),
    ]
    for mod, pkg, sub in packages:
        if check_import(mod, pkg, sub):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Transformers
    print("\n[3/8] Transformers & NLP:")
    packages = [
        ("transformers", None, None),
        ("tokenizers", None, None),
        ("qwen_vl_utils", "qwen-vl-utils", None),
        ("sentencepiece", None, None),
    ]
    for mod, pkg, sub in packages:
        if check_import(mod, pkg, sub):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Detection models
    print("\n[4/8] Detection & Segmentation:")
    packages = [
        ("supervision", None, None),
        ("groundingdino", "groundingdino-py", None),
        ("segment_anything", "segment-anything", None),
    ]
    for mod, pkg, sub in packages:
        if check_import(mod, pkg, sub):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Image processing
    print("\n[5/8] Image Processing:")
    packages = [
        ("PIL", "Pillow", None),
        ("cv2", "opencv-python-headless", None),
        ("numpy", None, None),
        ("scipy", None, None),
    ]
    for mod, pkg, sub in packages:
        if check_import(mod, pkg, sub):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # HuggingFace
    print("\n[6/8] HuggingFace Hub:")
    packages = [
        ("huggingface_hub", "huggingface-hub", None),
    ]
    for mod, pkg, sub in packages:
        if check_import(mod, pkg, sub):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # CLIP models
    print("\n[7/8] CLIP Models:")
    packages = [
        ("open_clip", "open-clip-torch", None),
    ]
    for mod, pkg, sub in packages:
        if check_import(mod, pkg, sub):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Image enhancement
    print("\n[8/8] Image Enhancement:")
    packages = [
        ("basicsr", None, None),
        ("realesrgan", None, None),
    ]
    for mod, pkg, sub in packages:
        if check_import(mod, pkg, sub):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Utilities
    print("\n[Bonus] Utilities:")
    packages = [
        ("tqdm", None, None),
        ("IPython", "ipython", None),
    ]
    for mod, pkg, sub in packages:
        if check_import(mod, pkg, sub):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    total = results["passed"] + results["failed"]
    print(f"Passed: {results['passed']}/{total}")
    print(f"Failed: {results['failed']}/{total}")
    
    if results["failed"] > 0:
        print("\n⚠ Some packages are missing. Install them with:")
        print("  pip install -r requirements.txt")
        print("\nFor CUDA support, install PyTorch separately:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return 1
    else:
        print("\n✓ All packages verified successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
