#!/usr/bin/env python3
"""
Pre-download ALL models for the synthetic pipeline.
Run this separately before running the main pipeline.
Downloads models one at a time to avoid VRAM issues.
"""
import os
import sys
import gc
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
print("MODEL DOWNLOADER - ALL PIPELINE MODELS")
print("="*70)

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

def get_free_memory():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        return free / (1024**3), total / (1024**3)
    return 0, 0

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Models to download
models = [
    {
        "name": "Qwen2-VL-2B-Instruct",
        "id": "Qwen/Qwen2-VL-2B-Instruct",
        "type": "transformers",
        "class": "Qwen2VLForConditionalGeneration"
    },
    {
        "name": "Grounding DINO",
        "id": "IDEA-Research/grounding-dino-base",
        "type": "transformers",
        "class": "AutoModelForZeroShotObjectDetection"
    },
    {
        "name": "SAM ViT-Huge",
        "id": "facebook/sam-vit-huge",
        "type": "transformers",
        "class": "SamModel"
    },
    {
        "name": "Stable Diffusion 3.5 Medium",
        "id": "stabilityai/stable-diffusion-3.5-medium",
        "type": "diffusers",
        "class": "StableDiffusion3Pipeline"
    }
]

print(f"Will download {len(models)} models sequentially")
print("(CLIP removed - quality scoring disabled for LoRA training)")
print()
print("NOTE: SD 3.5 is GATED - you must accept license first:")
print("  https://huggingface.co/stabilityai/stable-diffusion-3.5-medium")
print()

for i, model in enumerate(models, 1):
    print("="*70)
    print(f"[{i}/{len(models)}] {model['name']}")
    print("="*70)
    
    free, total = get_free_memory()
    print(f"Free GPU Memory: {free:.1f}/{total:.1f} GB")
    
    try:
        if model['type'] == 'transformers':
            print(f"Downloading from HuggingFace: {model['id']}")
            
            if 'Qwen' in model['class']:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                print("  - Downloading processor...")
                processor = AutoProcessor.from_pretrained(
                    model['id'],
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                print("  - Downloading model...")
                mdl = Qwen2VLForConditionalGeneration.from_pretrained(
                    model['id'],
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    cache_dir=cache_dir
                )
                del processor, mdl
                
            elif 'ZeroShot' in model['class']:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                print("  - Downloading processor...")
                processor = AutoProcessor.from_pretrained(
                    model['id'],
                    cache_dir=cache_dir
                )
                print("  - Downloading model...")
                mdl = AutoModelForZeroShotObjectDetection.from_pretrained(
                    model['id'],
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    cache_dir=cache_dir
                )
                del processor, mdl
                
            elif 'Sam' in model['class']:
                from transformers import SamModel, SamProcessor
                print("  - Downloading processor...")
                processor = SamProcessor.from_pretrained(
                    model['id'],
                    cache_dir=cache_dir
                )
                print("  - Downloading model...")
                mdl = SamModel.from_pretrained(
                    model['id'],
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    cache_dir=cache_dir
                )
                del processor, mdl
        
        elif model['type'] == 'diffusers':
            from diffusers import StableDiffusion3Pipeline
            print(f"Downloading from HuggingFace: {model['id']}")
            print("  - Downloading pipeline (this is large ~10GB)...")
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                model['id'],
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir=cache_dir
            )
            del pipeline
        
        print(f"✓ {model['name']} downloaded successfully!")
        
    except Exception as e:
        print(f"✗ Failed to download {model['name']}: {e}")
        import traceback
        traceback.print_exc()
        print("\nContinuing with next model...")
    
    # Clear memory before next model
    print("  - Clearing GPU memory...")
    clear_memory()
    
    free_after, _ = get_free_memory()
    print(f"  - Free GPU Memory after cleanup: {free_after:.1f} GB")
    print()

print("="*70)
print("DOWNLOAD COMPLETE")
print("="*70)
print("✓ All models cached!")
print(f"✓ Location: {cache_path}")
print()
print("You can now run run_stage1.py")
print("Models will load much faster from cache!")
print("="*70)
