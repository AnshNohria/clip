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
    python run_stage1.py --images-dir path/to/images --target-count 100
"""
import os
import sys
import argparse
from pathlib import Path


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
        default="outputs",
        help="Output directory for synthetic images and metadata"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=100,
        help="Target number of synthetic samples to generate (default: 100)"
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
    
    # Determine the project root
    script_dir = Path(__file__).resolve().parent
    
    # Default images directory
    if args.images_dir is None:
        images_dir = script_dir / "datasets" / "rsicd_images"
    else:
        images_dir = Path(args.images_dir)
    
    # Validate images directory
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        print("\nPlease ensure the rsicd_images folder exists at:")
        print(f"  {images_dir}")
        print("\nOr specify a custom path with --images-dir")
        return 1
    
    # Count available images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"ERROR: No images found in {images_dir}")
        return 1
    
    print("=" * 70)
    print("STAGE 1: SYNTHETIC GENERATION PIPELINE")
    print("=" * 70)
    print(f"Images directory: {images_dir}")
    print(f"Found {len(image_files)} images")
    print(f"Output directory: {args.output_dir}")
    print(f"Target count: {args.target_count}")
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
    
    # Configure the pipeline
    synthetic_config = SyntheticConfig()
    synthetic_config.target_synthetic_count = args.target_count
    synthetic_config.min_clip_score = args.min_clip_score
    synthetic_config.min_layout_iou = args.min_layout_iou
    
    config = PipelineConfig(
        device=args.device,
        output_dir=Path(args.output_dir) / "remoteclip_pipeline",
        synthetic=synthetic_config
    )
    
    # Create and run the pipeline
    print("\nInitializing Synthetic Generation Pipeline...")
    pipeline = SyntheticGenerationPipeline(config)
    
    try:
        print("\nLoading models (this may take a few minutes)...")
        pipeline.setup()
        
        print(f"\nProcessing images from: {images_dir}")
        print(f"Output will be saved to: {pipeline.synthetic_dir}")
        print(f"Metadata will be saved to: {pipeline.metadata_dir}")
        
        result = pipeline.generate_dataset(
            source_images_dir=images_dir,
            output_dir=Path(args.output_dir),
            target_count=args.target_count
        )
        
        print("\n" + "=" * 70)
        print("STAGE 1 COMPLETE")
        print("=" * 70)
        print(f"Generated samples: {result.get('generated', 0)}")
        print(f"Quality rate: {result.get('quality_rate', 0) * 100:.1f}%")
        print(f"Manifest saved to: {result.get('manifest_path', 'N/A')}")
        print(f"Images saved to: {pipeline.synthetic_dir}")
        print(f"Metadata saved to: {pipeline.metadata_dir}")
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
