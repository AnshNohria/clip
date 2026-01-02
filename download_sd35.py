#!/usr/bin/env python3
"""
Pre-download Stable Diffusion 3.5 Medium model.
Run this separately before running the main pipeline.
"""
import os
import sys
import torch
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Login to HuggingFace
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    print("ERROR: HF_TOKEN not found in .env file")
    sys.exit(1)

try:
    from huggingface_hub import login
    login(token=hf_token)
    print("✓ Logged in to HuggingFace")
except Exception as e:
    print(f"ERROR: Failed to login to HuggingFace: {e}")
    sys.exit(1)

# Check CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

print("\n" + "="*70)
print("SD 3.5 MODEL DOWNLOADER")
print("="*70)

# Get free memory
free_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
print(f"Free GPU Memory: {free_gb:.1f} GB")

if free_gb < 8:
    print(f"\nWARNING: Only {free_gb:.1f}GB free. Recommend at least 8GB for download.")
    print("Other models may be loaded. Consider restarting kernel/session.")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(0)

# Get cache directory - use absolute path
cache_dir = os.getenv('HF_HOME')
if not cache_dir:
    # Default to absolute path in project
    cache_dir = str(Path.cwd() / 'checkpoints' / 'huggingface')

cache_path = Path(cache_dir).resolve()
print(f"Cache directory: {cache_path}")

# Create cache directory if it doesn't exist
cache_path.mkdir(parents=True, exist_ok=True)
print()

# Model to download
model_id = "stabilityai/stable-diffusion-3.5-medium"
print(f"Downloading: {model_id}")
print("This will download ~10GB of model files...")
print()
print("NOTE: SD 3.5 is a GATED model. You must:")
print("1. Go to: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium")
print("2. Click 'Agree and access repository' (requires HF account)")
print("3. Wait a few minutes for access to be granted")
print()

try:
    from diffusers import StableDiffusion3Pipeline
    
    print("Step 1: Downloading model files...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=cache_dir
    )
    
    print("\n✓ Model files downloaded successfully!")
    print(f"✓ Cached at: {cache_path}")
    
    # Clear from memory
    del pipeline
    torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("SD 3.5 READY")
    print("="*70)
    print("You can now run run_stage1.py")
    print("The model will load much faster from cache!")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    
    if "401" in str(e) or "gated" in str(e).lower():
        print("\n" + "="*70)
        print("LICENSE ACCEPTANCE REQUIRED")
        print("="*70)
        print("SD 3.5 Medium is a gated model. Follow these steps:")
        print()
        print("1. Go to: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium")
        print("2. Log in with your HuggingFace account")
        print("3. Click the 'Agree and access repository' button")
        print("4. Wait 2-3 minutes for access approval")
        print("5. Run this script again")
        print()
        print("Make sure your HF_TOKEN in .env matches the account you used!")
        print("="*70)
    else:
        import traceback
        traceback.print_exc()
    
    sys.exit(1)
