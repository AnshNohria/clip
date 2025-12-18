"""
RemoteCLIP Synthetic-to-Real Transfer Learning Pipeline

A single-GPU pipelined training system with iterative refinement for 
generating perfect image-text pairs and fine-tuning RemoteCLIP.

Pipeline Stages:
    1A. Synthetic Generation (10-stage pipeline)
    1B. Zoom Crops Extraction (5-stage pipeline)
    2.  Synthetic-Heavy Training (80% synthetic, 20% crops)
    3.  Real-Dominated Training (85% crops, 15% synthetic)
    4.  Evaluation and Retrieval

Usage:
    from remoteclip_pipeline import RemoteCLIPPipeline, PipelineConfig
    
    config = PipelineConfig()
    pipeline = RemoteCLIPPipeline(config)
    pipeline.run_full_pipeline(real_images_dir=Path("./images"))
"""

__version__ = "1.0.0"
__author__ = "RemoteCLIP Pipeline"

from .config import (
    PipelineConfig, 
    SyntheticConfig, 
    ZoomCropConfig,
    TrainingConfig,
    EvaluationConfig
)
from .synthetic_pipeline import SyntheticGenerationPipeline
from .zoom_crops_pipeline import ZoomCropsPipeline
from .prompt_refinement import PromptRefinementEngine
from .stage2_trainer import Stage2SyntheticHeavyTrainer
from .stage3_trainer import Stage3RealDominatedTrainer
from .evaluation import RemoteCLIPEvaluator, ImageRetrievalIndex
from .fourier_augment import FourierAmplitudeSwap, FourierMixUp, AdaptiveFourierAugment
from .main import RemoteCLIPPipeline

__all__ = [
    # Config
    "PipelineConfig",
    "SyntheticConfig",
    "ZoomCropConfig",
    "TrainingConfig",
    "EvaluationConfig",
    # Pipelines
    "SyntheticGenerationPipeline",
    "ZoomCropsPipeline",
    "PromptRefinementEngine",
    # Trainers
    "Stage2SyntheticHeavyTrainer",
    "Stage3RealDominatedTrainer",
    # Evaluation
    "RemoteCLIPEvaluator",
    "ImageRetrievalIndex",
    # Augmentation
    "FourierAmplitudeSwap",
    "FourierMixUp",
    "AdaptiveFourierAugment",
    # Main
    "RemoteCLIPPipeline",
]
