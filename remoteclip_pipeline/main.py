#!/usr/bin/env python3
"""
Main Orchestrator: RemoteCLIP Synthetic-to-Real Training Pipeline

Complete end-to-end pipeline for training RemoteCLIP using
synthetic data generation and iterative refinement.

Usage:
    python -m remoteclip_pipeline.main --config config.yaml
    python -m remoteclip_pipeline.main --stage synthetic --real-images /path/to/images
    python -m remoteclip_pipeline.main --stage all
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies
torch: Any = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import (
    PipelineConfig, SyntheticConfig, ZoomCropConfig, 
    TrainingConfig, EvaluationConfig
)
from .synthetic_pipeline import SyntheticGenerationPipeline
from .zoom_crops_pipeline import ZoomCropsPipeline
from .prompt_refinement import PromptRefinementEngine
from .stage2_trainer import Stage2SyntheticHeavyTrainer
from .stage3_trainer import Stage3RealDominatedTrainer
from .evaluation import RemoteCLIPEvaluator


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

class RemoteCLIPPipeline:
    """
    Main orchestrator for the RemoteCLIP training pipeline.
    
    Stages:
    1A. Synthetic Generation (10-stage)
    1B. Zoom Crops Extraction (5-stage)
    2.  Synthetic-Heavy Training (80% synthetic)
    3.  Real-Dominated Training (85% crops)
    4.  Evaluation
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Stage controllers
        self.synthetic_pipeline: Optional[SyntheticGenerationPipeline] = None
        self.zoom_pipeline: Optional[ZoomCropsPipeline] = None
        self.prompt_engine: Optional[PromptRefinementEngine] = None
        self.stage2_trainer: Optional[Stage2SyntheticHeavyTrainer] = None
        self.stage3_trainer: Optional[Stage3RealDominatedTrainer] = None
        self.evaluator: Optional[RemoteCLIPEvaluator] = None
        
        # Paths
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.synthetic_dir = self.output_dir / "synthetic"
        self.crops_dir = self.output_dir / "crops"
        self.checkpoint_dir = self.config.checkpoint_dir
        
        # State
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stage_results: Dict[str, Any] = {}
        self.log_file = self.output_dir / f"pipeline_log_{self.run_id}.json"
    
    def log_stage(self, stage: str, status: str, metrics: Optional[Dict] = None):
        """Log stage completion."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'status': status,
            'metrics': metrics or {}
        }
        
        self.stage_results[stage] = entry
        
        # Append to log file
        log_data = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
        
        log_data.append(entry)
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"  [LOG] Stage '{stage}' completed with status: {status}")
    
    # ========================================================================
    # STAGE 1A: SYNTHETIC GENERATION
    # ========================================================================
    
    def run_synthetic_generation(
        self,
        real_images_dir: Path,
        target_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Stage 1A: 10-stage synthetic generation pipeline.
        
        Args:
            real_images_dir: Directory with real satellite images
            target_count: Target number of synthetic samples
        
        Returns:
            Generation results and statistics
        """
        print("\n" + "="*80)
        print("STAGE 1A: SYNTHETIC GENERATION (10-STAGE PIPELINE)")
        print("="*80)
        
        start_time = time.time()
        
        try:
            self.synthetic_pipeline = SyntheticGenerationPipeline(self.config)
            self.synthetic_pipeline.setup()
            
            target = target_count or self.config.synthetic.target_count
            
            # Force source directory to RS-TransCLIP/datasets/rsicd_images
            repo_root = Path(__file__).resolve().parent.parent
            forced_source = repo_root / "RS-TransCLIP" / "datasets" / "rsicd_images"
            if not forced_source.exists():
                raise FileNotFoundError(f"Required input directory missing: {forced_source}")
            
            result = self.synthetic_pipeline.generate_dataset(
                source_images_dir=forced_source,
                output_dir=self.synthetic_dir,
                target_count=target
            )
            
            elapsed = time.time() - start_time
            
            metrics = {
                'generated_count': result.get('generated', 0),
                'quality_rate': result.get('quality_rate', 0),
                'elapsed_hours': elapsed / 3600
            }
            
            self.log_stage("synthetic_generation", "success", metrics)
            
            return result
            
        except Exception as e:
            self.log_stage("synthetic_generation", "failed", {'error': str(e)})
            raise
    
    # ========================================================================
    # STAGE 1B: ZOOM CROPS EXTRACTION
    # ========================================================================
    
    def run_zoom_crops_extraction(
        self,
        images_dir: Path,
        target_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Stage 1B: 5-stage zoom crops extraction.
        
        Args:
            images_dir: Directory with satellite images
            target_count: Target number of crops
        
        Returns:
            Extraction results
        """
        print("\n" + "="*80)
        print("STAGE 1B: ZOOM CROPS EXTRACTION (5-STAGE PIPELINE)")
        print("="*80)
        
        start_time = time.time()
        
        try:
            self.zoom_pipeline = ZoomCropsPipeline(self.config)
            self.zoom_pipeline.setup()
            
            target = target_count or self.config.zoom_crops.target_crop_count
            
            result = self.zoom_pipeline.extract_all_crops(
                images_dir=images_dir,
                output_dir=self.crops_dir,
                target_count=target
            )
            
            elapsed = time.time() - start_time
            
            metrics = {
                'extracted_count': result.get('total_crops', 0),
                'images_processed': result.get('images_processed', 0),
                'elapsed_hours': elapsed / 3600
            }
            
            self.log_stage("zoom_crops", "success", metrics)
            
            return result
            
        except Exception as e:
            self.log_stage("zoom_crops", "failed", {'error': str(e)})
            raise
    
    # ========================================================================
    # STAGE 2: SYNTHETIC-HEAVY TRAINING
    # ========================================================================
    
    def run_stage2_training(self) -> Dict[str, Any]:
        """
        Run Stage 2: Synthetic-heavy training (80% synthetic, 20% crops).
        
        Returns:
            Training results
        """
        print("\n" + "="*80)
        print("STAGE 2: SYNTHETIC-HEAVY TRAINING")
        print("="*80)
        
        start_time = time.time()
        
        try:
            self.stage2_trainer = Stage2SyntheticHeavyTrainer(self.config)
            
            # Dataset paths
            synthetic_path = self.synthetic_dir / "dataset.json"
            crops_path = self.crops_dir / "dataset.json"
            
            result = self.stage2_trainer.train(
                synthetic_dataset_path=synthetic_path,
                crops_dataset_path=crops_path
            )
            
            elapsed = time.time() - start_time
            
            metrics = {
                'total_steps': result.get('total_steps', 0),
                'final_loss': result.get('final_loss', 0),
                'elapsed_hours': elapsed / 3600
            }
            
            self.log_stage("stage2_training", "success", metrics)
            
            return result
            
        except Exception as e:
            self.log_stage("stage2_training", "failed", {'error': str(e)})
            raise
    
    # ========================================================================
    # STAGE 3: REAL-DOMINATED TRAINING
    # ========================================================================
    
    def run_stage3_training(
        self,
        rsicd_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run Stage 3: Real-dominated training (85% crops, 15% synthetic).
        
        Args:
            rsicd_path: Path to RSICD test set for validation
        
        Returns:
            Training results
        """
        print("\n" + "="*80)
        print("STAGE 3: REAL-DOMINATED TRAINING")
        print("="*80)
        
        start_time = time.time()
        
        try:
            self.stage3_trainer = Stage3RealDominatedTrainer(self.config)
            
            # Dataset paths
            synthetic_path = self.synthetic_dir / "dataset.json"
            crops_path = self.crops_dir / "dataset.json"
            
            # RSICD path for validation
            if rsicd_path is None:
                rsicd_path = Path("RS-TransCLIP/datasets/rsicd_images")
            
            result = self.stage3_trainer.train(
                synthetic_path=synthetic_path,
                crops_path=crops_path,
                rsicd_path=rsicd_path
            )
            
            elapsed = time.time() - start_time
            
            metrics = {
                'best_r10': result.get('best_r10', 0),
                'total_steps': result.get('total_steps', 0),
                'elapsed_hours': elapsed / 3600
            }
            
            self.log_stage("stage3_training", "success", metrics)
            
            return result
            
        except Exception as e:
            self.log_stage("stage3_training", "failed", {'error': str(e)})
            raise
    
    # ========================================================================
    # STAGE 4: EVALUATION
    # ========================================================================
    
    def run_evaluation(
        self,
        test_images_dir: Path,
        checkpoint_path: Optional[Path] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run full evaluation.
        
        Args:
            test_images_dir: Directory with test images
            checkpoint_path: Model checkpoint to evaluate
            class_names: Class names for zero-shot classification
        
        Returns:
            Evaluation results
        """
        print("\n" + "="*80)
        print("STAGE 4: EVALUATION")
        print("="*80)
        
        start_time = time.time()
        
        try:
            self.evaluator = RemoteCLIPEvaluator(self.config)
            
            # Use best checkpoint by default
            if checkpoint_path is None:
                checkpoint_path = self.checkpoint_dir / "stage3" / "lora_best.pt"
            
            self.evaluator.load_model(checkpoint_path)
            
            # Default test queries for remote sensing
            test_queries = [
                "a dense residential area with many buildings",
                "an airport runway with aircraft",
                "a river winding through green fields",
                "a stadium in an urban area",
                "agricultural fields with crops",
                "a harbor with boats and ships",
                "a highway interchange",
                "a solar farm with panels",
                "a bridge crossing water",
                "a forest with dense tree coverage"
            ]
            
            result = self.evaluator.full_evaluation(
                image_dir=test_images_dir,
                test_queries=test_queries,
                class_names=class_names
            )
            
            elapsed = time.time() - start_time
            
            metrics = {
                'intra_class_sim': result.get('similarity', {}).get('intra_class_similarity', 0),
                'inter_class_sim': result.get('similarity', {}).get('inter_class_similarity', 0),
                'zero_shot_acc': result.get('zero_shot', {}).get('accuracy', 0),
                'elapsed_minutes': elapsed / 60
            }
            
            self.log_stage("evaluation", "success", metrics)
            
            return result
            
        except Exception as e:
            self.log_stage("evaluation", "failed", {'error': str(e)})
            raise
    
    # ========================================================================
    # FULL PIPELINE
    # ========================================================================
    
    def run_full_pipeline(
        self,
        real_images_dir: Path,
        rsicd_path: Optional[Path] = None,
        class_names: Optional[List[str]] = None,
        skip_generation: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete pipeline end-to-end.
        
        Args:
            real_images_dir: Directory with real satellite images
            rsicd_path: Path to RSICD for validation
            class_names: Class names for evaluation
            skip_generation: Skip synthetic/crop generation (use existing)
        
        Returns:
            Complete pipeline results
        """
        print("\n" + "="*80)
        print("REMOTECLIP SYNTHETIC-TO-REAL TRAINING PIPELINE")
        print("="*80)
        print(f"Run ID: {self.run_id}")
        print(f"Output: {self.output_dir}")
        print(f"Device: {self.config.device}")
        print("="*80)
        
        total_start = time.time()
        results = {}
        
        try:
            # Stage 1A: Synthetic Generation
            if not skip_generation:
                results['synthetic'] = self.run_synthetic_generation(real_images_dir)
                
                # Stage 1B: Zoom Crops
                results['crops'] = self.run_zoom_crops_extraction(real_images_dir)
            else:
                print("\n[SKIP] Using existing synthetic data and crops")
            
            # Stage 2: Synthetic-Heavy Training
            results['stage2'] = self.run_stage2_training()
            
            # Stage 3: Real-Dominated Training
            results['stage3'] = self.run_stage3_training(rsicd_path)
            
            # Stage 4: Evaluation
            results['evaluation'] = self.run_evaluation(
                test_images_dir=real_images_dir,
                class_names=class_names
            )
            
            total_elapsed = time.time() - total_start
            
            # Final summary
            print("\n" + "="*80)
            print("PIPELINE COMPLETE")
            print("="*80)
            print(f"Total time: {total_elapsed/3600:.2f} hours")
            print(f"Stage 2 final loss: {results.get('stage2', {}).get('final_loss', 'N/A')}")
            print(f"Stage 3 best R@10: {results.get('stage3', {}).get('best_r10', 'N/A'):.3f}")
            print(f"Zero-shot accuracy: {results.get('evaluation', {}).get('zero_shot', {}).get('accuracy', 'N/A')}")
            print(f"Results saved to: {self.log_file}")
            
            return results
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            raise


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RemoteCLIP Synthetic-to-Real Training Pipeline"
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "synthetic", "crops", "stage2", "stage3", "eval"],
        default="all",
        help="Pipeline stage to run"
    )
    
    parser.add_argument(
        "--real-images",
        type=str,
        required=False,
        default=None,
        help="Directory with real satellite images (ignored; pipeline uses RS-TransCLIP/datasets/rsicd_images)"
    )
    
    parser.add_argument(
        "--rsicd-path",
        type=str,
        default=None,
        help="Path to RSICD dataset for validation"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/remoteclip_pipeline",
        help="Output directory"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/remoteclip",
        help="Checkpoint directory"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=5000,
        help="Target synthetic sample count"
    )
    
    parser.add_argument(
        "--crop-count",
        type=int,
        default=50000,
        help="Target crop count"
    )
    
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip synthetic/crop generation"
    )
    
    return parser.parse_args()


def _configure_hf_cache(repo_root: Path):
    """Ensure Hugging Face cache resides in repo-local hf_cache unless user overrides."""
    cache_dir = repo_root / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(cache_dir)
    if not os.environ.get("HUGGINGFACE_HUB_CACHE"):
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)


def main():
    """Main entry point."""
    args = parse_args()

    # Set HF cache to repo-local folder if not provided
    repo_root = Path(__file__).resolve().parent.parent
    _configure_hf_cache(repo_root)
    
    # Create config
    synthetic_cfg = SyntheticConfig()
    synthetic_cfg.target_synthetic_count = args.synthetic_count
    
    zoom_cfg = ZoomCropConfig()
    zoom_cfg.target_crop_count = args.crop_count
    
    config = PipelineConfig(
        device=args.device,
        output_dir=Path(args.output_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        synthetic=synthetic_cfg,
        zoom_crops=zoom_cfg
    )
    
    # Create pipeline
    pipeline = RemoteCLIPPipeline(config)
    
    # Always use RS-TransCLIP/datasets/rsicd_images as input source
    fixed_real_dir = repo_root / "RS-TransCLIP" / "datasets" / "rsicd_images"
    real_images_dir = fixed_real_dir
    rsicd_path = Path(args.rsicd_path) if args.rsicd_path else None
    
    # Run selected stage
    if args.stage == "all":
        pipeline.run_full_pipeline(
            real_images_dir=real_images_dir,
            rsicd_path=rsicd_path,
            skip_generation=args.skip_generation
        )
    elif args.stage == "synthetic":
        pipeline.run_synthetic_generation(
            real_images_dir=real_images_dir,
            target_count=args.synthetic_count
        )
    elif args.stage == "crops":
        pipeline.run_zoom_crops_extraction(
            images_dir=real_images_dir,
            target_count=args.crop_count
        )
    elif args.stage == "stage2":
        pipeline.run_stage2_training()
    elif args.stage == "stage3":
        pipeline.run_stage3_training(rsicd_path)
    elif args.stage == "eval":
        pipeline.run_evaluation(test_images_dir=real_images_dir)


if __name__ == "__main__":
    main()
