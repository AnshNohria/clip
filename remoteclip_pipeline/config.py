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
    """Configuration for synthetic data generation pipeline."""
    
    # Real-ESRGAN settings
    esrgan_scale: int = 4
    esrgan_batch_size: int = 4
    esrgan_model: str = "realesrgan-x4plus"
    
    # Qwen2-VL settings
    qwen_model: str = "Qwen/Qwen2-VL-2B-Instruct"  # 2B model for memory efficiency
    qwen_max_tokens: int = 512
    
    # Grounding DINO settings
    gdino_model: str = "IDEA-Research/grounding-dino-base"
    gdino_batch_size: int = 16
    gdino_confidence: float = 0.35
    gdino_box_threshold: float = 0.35
    gdino_text_threshold: float = 0.25
    
    # SAM settings
    sam_model: str = "facebook/sam-vit-huge"
    sam_checkpoint: Optional[str] = None
    
    # SD 3.5 settings
    sd_model: str = "stabilityai/stable-diffusion-3.5-large"
    sd_steps: int = 50
    sd_guidance_scale: float = 7.5
    sd_controlnet_scale: float = 0.8
    sd_image_size: int = 1024
    
    # Quality thresholds
    min_clip_score: float = 0.75
    min_layout_iou: float = 0.6
    min_object_count_accuracy: float = 0.8
    target_quality_rate: float = 0.85
    
    # Queue settings
    queue_buffer_depth: int = 32
    
    # Output settings
    target_synthetic_count: int = 4500
    save_intermediate: bool = True
    
    @property
    def target_count(self) -> int:
        """Alias for target_synthetic_count."""
        return self.target_synthetic_count


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
