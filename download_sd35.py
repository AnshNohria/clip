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

# Check CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

print("="*70)
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

# Get cache directory
cache_dir = os.getenv('HF_HOME', 'checkpoints/huggingface')
cache_path = Path(cache_dir).resolve()
print(f"Cache directory: {cache_path}")
print()

# Model to download
model_id = "stabilityai/stable-diffusion-3.5-medium"
print(f"Downloading: {model_id}")
print("This will download ~10GB of model files...")
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
    import traceback
    traceback.print_exc()
    sys.exit(1)
