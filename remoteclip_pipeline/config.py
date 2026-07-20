#!/usr/bin/env python3
"""
Configuration classes for the RemoteCLIP pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation pipeline.

    Tuned for 2x NVIDIA A100 (80GB) + 7 CPU cores. The generator model
    (image synthesis) is placed on `gen_device` and the analysis models
    (VLM captioner + detector) are placed on `analysis_device`, so both
    GPUs stay busy across the per-image loop.

    Fan-out: for each source image, `n_prompts_per_image` prompt variants
    are generated -> that many synthetic images -> each image gets
    `n_captions_per_image` distinct captions. So:
        images_generated  = n_source_images * n_prompts_per_image
        pairs_generated   = images_generated * n_captions_per_image
    Defaults below (10,000 source images x 5 prompts x 5 captions) yield
    50,000 images / 250,000 image-caption pairs.
    """

    # Device placement (multi-GPU)
    gen_device: str = "cuda:0"        # image generator (heaviest single model)
    analysis_device: str = "cuda:1"   # VLM captioner + detector

    # Qwen2.5-VL settings (captioning / scene analysis)
    qwen_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    qwen_dtype: str = "bfloat16"
    qwen_max_tokens: int = 512

    # Grounding DINO settings (object/spatial grounding for prompts+captions)
    gdino_model: str = "IDEA-Research/grounding-dino-base"
    gdino_dtype: str = "bfloat16"
    gdino_confidence: float = 0.35
    gdino_box_threshold: float = 0.35
    gdino_text_threshold: float = 0.25

    # Image generator settings (FLUX.1-dev preferred; SD3.5-Large fallback)
    sd_model: str = "black-forest-labs/FLUX.1-dev"
    sd_backend: str = "flux"  # "flux" | "sd3"
    sd_dtype: str = "bfloat16"
    sd_steps: int = 28
    sd_guidance_scale: float = 3.5
    sd_image_size: int = 1024

    # Fan-out settings
    n_source_images: int = 10000
    n_prompts_per_image: int = 5
    n_captions_per_image: int = 5

    # Output settings
    target_synthetic_count: int = 50000  # = n_source_images * n_prompts_per_image
    save_intermediate: bool = True

    # CPU settings (7 cores available; leave 1 free for the OS/IO)
    cpu_threads: int = 6

    @property
    def target_count(self) -> int:
        """Alias for target_synthetic_count."""
        return self.target_synthetic_count

    @property
    def target_pair_count(self) -> int:
        """Total image-caption pairs = images * captions per image."""
        return self.target_synthetic_count * self.n_captions_per_image


@dataclass
class ZoomCropConfig:
    """Configuration for zoom crop extraction pipeline."""
    
    # Detection settings
    gdino_batch_size: int = 16
    gdino_confidence: float = 0.35
    
    # Crop settings
    top_k_crops: int = 10
    crop_padding: float = 0.20
    min_crop_size: int = 64
    max_crop_size: int = 512
    
    # Parallel processing
    num_workers: int = 8
    
    # Qwen captioning
    caption_max_words: int = 10
    
    # Output settings
    target_crop_count: int = 50000


@dataclass
class TrainingConfig:
    """Configuration for LoRA fine-tuning stages."""
    
    # Model settings
    backbone: str = "RemoteCLIP-ViT-B-32"
    remoteclip_checkpoint: Path = field(default_factory=lambda: Path("RemoteCLIP_checkpoints"))
    embed_dim: int = 512
    
    # LoRA settings
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Stage 2 settings (Synthetic-Heavy)
    stage2_synthetic_ratio: float = 0.80
    stage2_crop_ratio: float = 0.20
    stage2_batch_size: int = 256
    stage2_grad_accum_steps: int = 4  # Effective batch 1024
    stage2_learning_rate: float = 5e-4
    stage2_epochs: int = 15
    stage2_warmup_steps: int = 1000
    
    # Stage 3 settings (Real-Dominated)
    stage3_crop_ratio: float = 0.85
    stage3_synthetic_ratio: float = 0.15
    stage3_batch_size: int = 256
    stage3_grad_accum_steps: int = 4
    stage3_learning_rate: float = 1e-4
    stage3_epochs: int = 10
    stage3_warmup_steps: int = 500
    stage3_kl_weight: float = 0.1
    stage3_early_stopping_patience: int = 3
    
    # Loss settings
    temperature: float = 0.07
    
    # Data loading
    num_workers: int = 16
    prefetch_factor: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    checkpoint_interval: int = 1000
    async_checkpoint: bool = True
    
    # Mixed precision
    use_bf16: bool = True
    
    # Fourier augmentation
    fourier_probability: float = 0.30
    fourier_swap_ratio: float = 0.5


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and retrieval."""
    
    # RSICD test settings
    rsicd_test_path: Path = field(default_factory=lambda: Path("datasets/RSICD/test"))
    
    # Retrieval settings
    top_k: int = 10
    top_k_retrieval: int = 10  # Alias for top_k
    
    # Embedding quality targets
    intra_class_similarity_min: float = 0.7
    intra_class_similarity_max: float = 0.9
    inter_class_similarity_min: float = 0.2
    inter_class_similarity_max: float = 0.4
    
    # Batch settings
    eval_batch_size: int = 64


@dataclass
class PipelineConfig:
    """Master configuration for the entire pipeline."""
    
    # Sub-configurations
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    zoom_crops: ZoomCropConfig = field(default_factory=ZoomCropConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Global paths
    output_dir: Path = field(default_factory=lambda: Path("outputs/remoteclip_pipeline"))
    dataset_dir: Path = field(default_factory=lambda: Path("datasets/stage1_corpus"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/remoteclip"))
    
    # GPU settings
    device: str = "cuda"
    
    # Logging
    log_interval: int = 50
    verbose: bool = True
    
    def __post_init__(self):
        """Create directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories
        (self.output_dir / "synthetic").mkdir(exist_ok=True)
        (self.output_dir / "zoom_crops").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.checkpoint_dir / "stage2").mkdir(exist_ok=True)
        (self.checkpoint_dir / "stage3").mkdir(exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)
