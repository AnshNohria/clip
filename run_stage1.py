#!/usr/bin/env python3
"""
Stage 1 Runner: Synthetic Generation Pipeline

This script runs Stage 1 (Synthetic Generation) on the RSICD images folder.
Stage 1 includes the 10-stage synthetic generation pipeline:
    1. Real-ESRGAN upsampling (4x)
    2. Qwen2-VL dense scene analysis
    3. Grounding DINO layout detection
    4. SAM segmentation
    5. Intelligent Prompt Generator
    6. SD 3.5 generation with ControlNet
    7. Second-pass Qwen2-VL verification
    8. Second-pass Grounding DINO detection
    9. Second-pass SAM segmentation
    10. Final Prompt Refinement

Usage:
    python run_stage1.py
    python run_stage1.py --images-dir path/to/images
"""
import os
import sys
import argparse
from pathlib import Path


def load_env_file(env_path: Path):
    """Load environment variables from .env file."""
    if not env_path.exists():
        print(f"WARNING: .env file not found at {env_path}")
        return False
    
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
    return True


def setup_huggingface_auth(script_dir: Path):
    """Setup HuggingFace authentication and cache directory."""
    # Load .env file
    env_path = script_dir / ".env"
    load_env_file(env_path)
    
    # Setup checkpoints directory for HuggingFace cache
    hf_cache_dir = script_dir / "checkpoints" / "huggingface"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set HuggingFace environment variables to use local cache
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")
    
    # Check for HuggingFace token
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        print(f"✓ HuggingFace token loaded from .env")
    else:
        print("WARNING: No HuggingFace token found in .env file")
        print("  Some models (like SD 3.5) may require authentication")
        print("  Add HUGGINGFACE_HUB_TOKEN=your_token to .env file")
    
    print(f"✓ Model cache directory: {hf_cache_dir}")
    return hf_cache_dir


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 Synthetic Generation Pipeline")
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Path to input images directory (default: datasets/rsicd_images)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for synthetic images and metadata (default: outputs in clip folder)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu' (default: cuda)"
    )
    parser.add_argument(
        "--min-clip-score",
        type=float,
        default=0.75,
        help="Minimum CLIP score threshold (default: 0.75)"
    )
    parser.add_argument(
        "--min-layout-iou",
        type=float,
        default=0.6,
        help="Minimum layout IoU threshold (default: 0.6)"
    )
    
    args = parser.parse_args()
    
    # Determine the project root (clip folder)
    script_dir = Path(__file__).resolve().parent
    
    # Setup HuggingFace authentication and cache BEFORE importing any HF libraries
    print("=" * 70)
    print("SETTING UP ENVIRONMENT")
    print("=" * 70)
    hf_cache_dir = setup_huggingface_auth(script_dir)
    
    # Default images directory - datasets/rsicd_images in clip folder
    if args.images_dir is None:
        images_dir = script_dir / "datasets" / "rsicd_images"
    else:
        images_dir = Path(args.images_dir)
    
    # Output directory - outputs folder in clip folder
    if args.output_dir is None:
        output_dir = script_dir / "outputs"
    else:
        output_dir = Path(args.output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir = output_dir / "images"
    captions_output_dir = output_dir / "captions"
    metadata_output_dir = output_dir / "metadata"
    checkpoints_dir = script_dir / "checkpoints"
    
    images_output_dir.mkdir(parents=True, exist_ok=True)
    captions_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate images directory
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        print("\nPlease ensure the rsicd_images folder exists at:")
        print(f"  {images_dir}")
        print("\nOr specify a custom path with --images-dir")
        return 1
    
    # Count ALL available images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"ERROR: No images found in {images_dir}")
        return 1
    
    # Process ALL images - set target to total count
    total_images = len(image_files)
    
    print("=" * 70)
    print("STAGE 1: SYNTHETIC GENERATION PIPELINE")
    print("=" * 70)
    print(f"Images directory: {images_dir}")
    print(f"Found {total_images} images - will process ALL")
    print(f"Output directory: {output_dir}")
    print(f"  - Generated images: {images_output_dir}")
    print(f"  - Captions: {captions_output_dir}")
    print(f"  - Metadata: {metadata_output_dir}")
    print(f"Checkpoints directory: {checkpoints_dir}")
    print(f"Device: {args.device}")
    print(f"Quality thresholds: CLIP>{args.min_clip_score}, IoU>{args.min_layout_iou}")
    print("=" * 70)
    
    # Import pipeline components
    try:
        from remoteclip_pipeline.config import PipelineConfig, SyntheticConfig
        from remoteclip_pipeline.synthetic_pipeline import SyntheticGenerationPipeline
    except ImportError as e:
        print(f"ERROR: Failed to import pipeline modules: {e}")
        print("\nMake sure you're running from the project root directory")
        return 1
    
    # Configure the pipeline - process ALL images
    synthetic_config = SyntheticConfig()
    synthetic_config.target_synthetic_count = total_images  # Process ALL images
    synthetic_config.min_clip_score = args.min_clip_score
    synthetic_config.min_layout_iou = args.min_layout_iou
    
    config = PipelineConfig(
        device=args.device,
        output_dir=output_dir,
        checkpoint_dir=checkpoints_dir,
        synthetic=synthetic_config
    )
    
    # Create and run the pipeline
    print("\nInitializing Synthetic Generation Pipeline...")
    pipeline = SyntheticGenerationPipeline(config)
    
    # Override output directories to use our structure
    pipeline.synthetic_dir = images_output_dir
    pipeline.metadata_dir = metadata_output_dir
    
    try:
        print("\nLoading models (this may take a few minutes)...")
        pipeline.setup()
        
        print(f"\nProcessing ALL {total_images} images from: {images_dir}")
        print(f"Generated images will be saved to: {images_output_dir}")
        print(f"Metadata will be saved to: {metadata_output_dir}")
        
        result = pipeline.generate_dataset(
            source_images_dir=images_dir,
            output_dir=output_dir,
            target_count=total_images  # Process ALL images
        )
        
        # Save captions to separate file
        captions_file = captions_output_dir / "captions.txt"
        if result.get("samples"):
            with open(captions_file, "w", encoding="utf-8") as f:
                for sample in result["samples"]:
                    image_name = Path(sample.synthetic_image_path).name
                    caption = sample.refined_caption
                    f.write(f"{image_name}\t{caption}\n")
            print(f"Captions saved to: {captions_file}")
        
        print("\n" + "=" * 70)
        print("STAGE 1 COMPLETE")
        print("=" * 70)
        print(f"Total images processed: {total_images}")
        print(f"Successfully generated: {result.get('generated', 0)}")
        print(f"Quality rate: {result.get('quality_rate', 0) * 100:.1f}%")
        print(f"Manifest saved to: {result.get('manifest_path', 'N/A')}")
        print(f"Generated images: {images_output_dir}")
        print(f"Captions: {captions_output_dir}")
        print(f"Metadata: {metadata_output_dir}")
        print("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
