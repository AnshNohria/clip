#!/usr/bin/env python3
"""
STAGE 2: Synthetic-Heavy Pretraining / Fine-Tuning

This module implements LoRA fine-tuning on a frozen RemoteCLIP backbone
using the Stage 1 filtered dataset with a synthetic-heavy data mix.

TRAINING SETUP:
1. Backbone: Frozen RemoteCLIP (ViT-B/32)
2. Trainable: LoRA adapters + projection heads
3. Data mix: 70-90% synthetic + 10-30% real

LOSSES:
1. CLIP Contrastive Loss (primary)
2. Rare Class Weighting (upweight underrepresented categories)
3. Fine-Grained Word-Region Alignment (OPTIONAL)
   - Align word embeddings with spatial regions
   - Uses Grounding DINO detections from Stage 1

OUTPUT:
- lora_checkpoint.pt: LoRA adapter weights
- projection_heads.pt: Learned projection heads
- training_log.json: Loss curves and metrics
"""

import os
import sys
import json
import math
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Check for dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"
    print("⚠ PyTorch not available. Stage 2 requires GPU.")

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Stage2Config:
    """Configuration for Stage 2 LoRA fine-tuning"""
    
    # Data paths
    FILTERED_DATASET: Path = Path("datasets/stage1_corpus/filtered/filtered_dataset.json")
    OUTPUT_DIR: Path = Path("checkpoints/stage2_lora")
    
    # Data mixing
    SYNTHETIC_RATIO: float = 0.80  # 70-90% synthetic
    REAL_RATIO: float = 0.20  # 10-30% real
    
    # Model settings
    BACKBONE: str = "RemoteCLIP-ViT-B-32"  # Frozen backbone
    REMOTECLIP_CHECKPOINT: Path = Path("RemoteCLIP_checkpoints")
    EMBED_DIM: int = 512  # CLIP embedding dimension
    
    # LoRA settings
    LORA_RANK: int = 8  # LoRA rank (r)
    LORA_ALPHA: float = 16.0  # LoRA scaling factor
    LORA_DROPOUT: float = 0.1
    LORA_TARGET_MODULES: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "out_proj",  # Attention
        "fc1", "fc2"  # MLP
    ])
    
    # Training settings
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    NUM_EPOCHS: int = 10
    WARMUP_STEPS: int = 500
    GRAD_ACCUM_STEPS: int = 1
    
    # Loss settings
    TEMPERATURE: float = 0.07  # CLIP temperature
    RARE_CLASS_WEIGHT: float = 2.0  # Upweight factor for rare classes
    USE_FINE_GRAINED_LOSS: bool = True  # Enable word-region alignment
    FINE_GRAINED_WEIGHT: float = 0.3  # Weight for fine-grained loss
    
    # Rare class detection
    RARE_CLASS_THRESHOLD: int = 10  # Classes with fewer samples are "rare"
    
    # Logging
    LOG_INTERVAL: int = 50
    SAVE_INTERVAL: int = 500
    
    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LORA IMPLEMENTATION
# ============================================================================

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer.
    
    Adds trainable low-rank matrices A and B to a frozen linear layer:
    h = W₀x + BAx (where W₀ is frozen, A and B are learned)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.
        
        Args:
            x: Input tensor
            original_output: Output from frozen original layer
        
        Returns:
            Adapted output = original + LoRA(x)
        """
        # LoRA path: x -> A -> B -> scale
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)  # x @ A.T
        lora_out = F.linear(lora_out, self.lora_B)  # (x @ A.T) @ B.T
        lora_out = lora_out * self.scaling
        
        return original_output + lora_out


class LoRALinear(nn.Module):
    """
    Wrapper that adds LoRA to an existing nn.Linear layer.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_layer(x)
        return self.lora(x, original_out)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.
    
    Args:
        model: Base model (will be modified in-place)
        target_modules: List of module name patterns to apply LoRA to
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout rate
    
    Returns:
        Modified model with LoRA layers
    """
    lora_modules = []
    
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        should_apply = any(target in name for target in target_modules)
        
        if should_apply and isinstance(module, nn.Linear):
            # Create LoRA wrapper
            lora_linear = LoRALinear(
                original_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
            
            # Replace in parent
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            
            setattr(parent, child_name, lora_linear)
            lora_modules.append(name)
    
    print(f"Applied LoRA to {len(lora_modules)} modules")
    return model


# ============================================================================
# PROJECTION HEADS
# ============================================================================

class ProjectionHead(nn.Module):
    """
    Learned projection head for CLIP embeddings.
    
    Maps CLIP features to a new embedding space for fine-tuning.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=-1)


# ============================================================================
# DATASET
# ============================================================================

class Stage2Dataset(Dataset):
    """
    PyTorch Dataset for Stage 2 training.
    
    Handles data mixing (synthetic vs real) and returns
    image-caption pairs with metadata.
    """
    
    def __init__(
        self,
        dataset_path: Path,
        transform,
        tokenizer,
        synthetic_ratio: float = 0.8,
        max_text_length: int = 77
    ):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        self.pairs = data['pairs']
        self.transform = transform
        self.tokenizer = tokenizer
        self.synthetic_ratio = synthetic_ratio
        self.max_text_length = max_text_length
        
        # Separate by source
        self.real_pairs = [p for p in self.pairs if p['source'] in ['real', 'real_augmented']]
        self.synthetic_pairs = [p for p in self.pairs if p['source'] == 'synthetic']
        
        # Get dataset directory
        self.dataset_dir = dataset_path.parent.parent  # Go up from filtered/
        
        # Compute class frequencies for rare class weighting
        self.class_counts = self._compute_class_counts()
        
        print(f"Loaded dataset: {len(self.real_pairs)} real, {len(self.synthetic_pairs)} synthetic")
    
    def _compute_class_counts(self) -> Counter:
        """Count scene types / classes for weighting"""
        counts = Counter()
        
        for pair in self.pairs:
            # Extract scene type from caption or metadata
            caption = pair.get('caption', '').lower()
            
            # Simple keyword-based classification
            if 'urban' in caption or 'building' in caption or 'residential' in caption:
                counts['urban'] += 1
            elif 'vegetation' in caption or 'forest' in caption or 'tree' in caption:
                counts['vegetation'] += 1
            elif 'water' in caption or 'river' in caption or 'lake' in caption:
                counts['water'] += 1
            elif 'agricultural' in caption or 'farm' in caption or 'field' in caption:
                counts['agricultural'] += 1
            else:
                counts['other'] += 1
        
        return counts
    
    def get_class_weights(self, rare_threshold: int = 10, upweight_factor: float = 2.0) -> Dict[str, float]:
        """Get weights for rare classes"""
        weights = {}
        total = sum(self.class_counts.values())
        
        for cls, count in self.class_counts.items():
            if count < rare_threshold:
                weights[cls] = upweight_factor
            else:
                weights[cls] = 1.0
        
        return weights
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        # Sample according to mixing ratio
        if random.random() < self.synthetic_ratio and self.synthetic_pairs:
            pair = random.choice(self.synthetic_pairs)
        elif self.real_pairs:
            pair = random.choice(self.real_pairs)
        else:
            pair = self.pairs[idx]
        
        # Load image
        image_path = self.dataset_dir / pair['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # Return a dummy image on error
            image = torch.zeros(3, 224, 224)
        
        # Tokenize caption
        caption = pair['caption']
        text_tokens = self.tokenizer([caption], truncate=True)[0]
        
        # Get class for weighting
        caption_lower = caption.lower()
        if 'urban' in caption_lower or 'building' in caption_lower:
            scene_class = 'urban'
        elif 'vegetation' in caption_lower or 'forest' in caption_lower:
            scene_class = 'vegetation'
        elif 'water' in caption_lower:
            scene_class = 'water'
        elif 'agricultural' in caption_lower or 'farm' in caption_lower:
            scene_class = 'agricultural'
        else:
            scene_class = 'other'
        
        return {
            'image': image,
            'text': text_tokens,
            'caption': caption,
            'source': pair['source'],
            'scene_class': scene_class,
            'image_id': pair['image_id'],
            'detection_counts': pair.get('detection_counts'),
            'spatial_layout': pair.get('spatial_layout')
        }


# ============================================================================
# LOSSES
# ============================================================================

class CLIPContrastiveLoss(nn.Module):
    """
    Standard CLIP contrastive loss (InfoNCE).
    
    Maximizes similarity of matched image-text pairs while
    minimizing similarity of unmatched pairs.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1/temperature)))
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute CLIP loss.
        
        Args:
            image_features: (B, D) normalized image embeddings
            text_features: (B, D) normalized text embeddings
            sample_weights: (B,) optional per-sample weights
        
        Returns:
            loss, metrics dict
        """
        # Get temperature
        temperature = torch.exp(self.log_temperature).clamp(min=0.01, max=100)
        
        # Compute similarities
        logits = (image_features @ text_features.T) / temperature
        
        # Targets: diagonal is positive pair
        batch_size = image_features.shape[0]
        targets = torch.arange(batch_size, device=logits.device)
        
        # Cross-entropy in both directions
        loss_i2t = F.cross_entropy(logits, targets, reduction='none')
        loss_t2i = F.cross_entropy(logits.T, targets, reduction='none')
        
        # Apply sample weights if provided
        if sample_weights is not None:
            loss_i2t = loss_i2t * sample_weights
            loss_t2i = loss_t2i * sample_weights
        
        loss = (loss_i2t.mean() + loss_t2i.mean()) / 2
        
        # Compute accuracy
        with torch.no_grad():
            i2t_acc = (logits.argmax(dim=1) == targets).float().mean()
            t2i_acc = (logits.argmax(dim=0) == targets).float().mean()
        
        metrics = {
            'loss_i2t': loss_i2t.mean().item(),
            'loss_t2i': loss_t2i.mean().item(),
            'acc_i2t': i2t_acc.item(),
            'acc_t2i': t2i_acc.item(),
            'temperature': temperature.item()
        }
        
        return loss, metrics


class FineGrainedAlignmentLoss(nn.Module):
    """
    Fine-grained word-region alignment loss.
    
    Aligns word embeddings with spatial regions using attention.
    This is OPTIONAL and adds extra computation.
    """
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        
        # Cross-attention for word-region alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Projection for region features (if using spatial tokens)
        self.region_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(
        self,
        word_features: torch.Tensor,  # (B, num_words, D)
        image_features: torch.Tensor,  # (B, num_patches, D) or (B, D) for CLS
        region_masks: Optional[torch.Tensor] = None  # (B, num_regions)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute fine-grained alignment loss.
        
        If image_features is 2D (B, D), we skip detailed alignment
        and return a simple similarity loss.
        """
        # If we only have CLS token, do simple alignment
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)  # (B, 1, D)
        
        # Cross-attention: query=words, key/value=image patches
        attended, attention_weights = self.cross_attention(
            query=word_features,
            key=image_features,
            value=image_features
        )
        
        # Alignment loss: maximize attention on correct regions
        # For now, use a simple contrastive loss on attended features
        attended_pooled = attended.mean(dim=1)  # (B, D)
        image_pooled = image_features.mean(dim=1)  # (B, D)
        
        # Normalize
        attended_pooled = F.normalize(attended_pooled, dim=-1)
        image_pooled = F.normalize(image_pooled, dim=-1)
        
        # Cosine similarity loss
        similarity = (attended_pooled * image_pooled).sum(dim=-1)
        loss = 1 - similarity.mean()
        
        metrics = {
            'fine_grained_similarity': similarity.mean().item(),
            'attention_entropy': -(attention_weights * attention_weights.log().clamp(min=-100)).sum(dim=-1).mean().item()
        }
        
        return loss, metrics


# ============================================================================
# TRAINER
# ============================================================================

class Stage2Trainer:
    """
    Main trainer for Stage 2 LoRA fine-tuning.
    """
    
    def __init__(self, config: Stage2Config):
        self.config = config
        self.device = DEVICE
        
        # Will be initialized in setup()
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.projection_image = None
        self.projection_text = None
        self.optimizer = None
        self.scheduler = None
        self.clip_loss = None
        self.fine_grained_loss = None
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_log = []
    
    def setup(self):
        """Initialize model, optimizers, and data loaders"""
        print("\n" + "="*80)
        print("STAGE 2 SETUP: Loading Models and Data")
        print("="*80)
        
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for Stage 2 training")
        
        # 1. Load CLIP/RemoteCLIP model
        print("\n[1/5] Loading CLIP backbone...")
        self._load_clip_model()
        
        # 2. Apply LoRA
        print("\n[2/5] Applying LoRA adapters...")
        self._apply_lora()
        
        # 3. Create projection heads
        print("\n[3/5] Creating projection heads...")
        self._create_projection_heads()
        
        # 4. Setup optimizer and scheduler
        print("\n[4/5] Setting up optimizer...")
        self._setup_optimizer()
        
        # 5. Setup losses
        print("\n[5/5] Setting up loss functions...")
        self._setup_losses()
        
        print("\n✓ Setup complete")
    
    def _load_clip_model(self):
        """Load CLIP/RemoteCLIP backbone"""
        try:
            import open_clip
            
            # Try RemoteCLIP first
            checkpoint_dir = self.config.REMOTECLIP_CHECKPOINT
            checkpoints = list(checkpoint_dir.rglob("*.pt")) if checkpoint_dir.exists() else []
            checkpoints.extend(checkpoint_dir.rglob("*.bin") if checkpoint_dir.exists() else [])
            
            if checkpoints:
                print(f"  Loading RemoteCLIP from {checkpoints[0]}")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    pretrained=str(checkpoints[0])
                )
                self.model_type = "RemoteCLIP"
            else:
                # Try HuggingFace
                try:
                    from huggingface_hub import hf_hub_download
                    checkpoint_path = hf_hub_download(
                        repo_id="chendelong/RemoteCLIP",
                        filename="RemoteCLIP-ViT-B-32.pt"
                    )
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        'ViT-B-32',
                        pretrained=checkpoint_path
                    )
                    self.model_type = "RemoteCLIP"
                    print("  ✓ Loaded RemoteCLIP from HuggingFace")
                except:
                    # Fallback to OpenAI CLIP
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        'ViT-B-32',
                        pretrained='openai'
                    )
                    self.model_type = "OpenCLIP"
                    print("  ✓ Loaded OpenCLIP (fallback)")
            
            self.model = model.to(self.device)
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
        except ImportError:
            # Try OpenAI CLIP
            try:
                import clip
                model, preprocess = clip.load("ViT-B/32", device=self.device)
                self.model = model
                self.preprocess = preprocess
                self.tokenizer = clip.tokenize
                self.model_type = "OpenAI-CLIP"
                print("  ✓ Loaded OpenAI CLIP")
            except ImportError:
                raise RuntimeError("No CLIP implementation available. Install open_clip or clip.")
        
        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"  Model type: {self.model_type}")
        print(f"  Device: {self.device}")
    
    def _apply_lora(self):
        """Apply LoRA to the model"""
        # Apply to visual encoder
        if hasattr(self.model, 'visual'):
            self.model.visual = apply_lora_to_model(
                self.model.visual,
                target_modules=self.config.LORA_TARGET_MODULES,
                rank=self.config.LORA_RANK,
                alpha=self.config.LORA_ALPHA,
                dropout=self.config.LORA_DROPOUT
            )
        
        # Apply to text encoder (transformer)
        if hasattr(self.model, 'transformer'):
            self.model.transformer = apply_lora_to_model(
                self.model.transformer,
                target_modules=self.config.LORA_TARGET_MODULES,
                rank=self.config.LORA_RANK,
                alpha=self.config.LORA_ALPHA,
                dropout=self.config.LORA_DROPOUT
            )
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def _create_projection_heads(self):
        """Create trainable projection heads"""
        self.projection_image = ProjectionHead(
            input_dim=self.config.EMBED_DIM,
            hidden_dim=self.config.EMBED_DIM,
            output_dim=256,
            num_layers=2
        ).to(self.device)
        
        self.projection_text = ProjectionHead(
            input_dim=self.config.EMBED_DIM,
            hidden_dim=self.config.EMBED_DIM,
            output_dim=256,
            num_layers=2
        ).to(self.device)
        
        proj_params = sum(p.numel() for p in self.projection_image.parameters())
        proj_params += sum(p.numel() for p in self.projection_text.parameters())
        print(f"  Projection head params: {proj_params:,}")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Collect trainable parameters
        params = []
        
        # LoRA parameters from model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append({'params': param, 'name': name})
        
        # Projection head parameters
        params.append({'params': self.projection_image.parameters(), 'name': 'projection_image'})
        params.append({'params': self.projection_text.parameters(), 'name': 'projection_text'})
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Cosine scheduler with warmup
        def lr_lambda(step):
            if step < self.config.WARMUP_STEPS:
                return step / self.config.WARMUP_STEPS
            total_steps = self.config.NUM_EPOCHS * 1000  # Approximate
            progress = (step - self.config.WARMUP_STEPS) / (total_steps - self.config.WARMUP_STEPS)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        print(f"  Optimizer: AdamW (lr={self.config.LEARNING_RATE})")
        print(f"  Warmup steps: {self.config.WARMUP_STEPS}")
    
    def _setup_losses(self):
        """Setup loss functions"""
        self.clip_loss = CLIPContrastiveLoss(temperature=self.config.TEMPERATURE)
        self.clip_loss.to(self.device)
        
        if self.config.USE_FINE_GRAINED_LOSS:
            self.fine_grained_loss = FineGrainedAlignmentLoss(
                embed_dim=self.config.EMBED_DIM
            ).to(self.device)
            print("  ✓ Fine-grained alignment loss ENABLED")
        else:
            self.fine_grained_loss = None
            print("  ✓ Fine-grained alignment loss DISABLED")
    
    def create_dataloader(self) -> DataLoader:
        """Create training data loader"""
        dataset = Stage2Dataset(
            dataset_path=self.config.FILTERED_DATASET,
            transform=self.preprocess,
            tokenizer=self.tokenizer,
            synthetic_ratio=self.config.SYNTHETIC_RATIO
        )
        
        # Get class weights for rare class upweighting
        self.class_weights = dataset.get_class_weights(
            rare_threshold=self.config.RARE_CLASS_THRESHOLD,
            upweight_factor=self.config.RARE_CLASS_WEIGHT
        )
        
        print(f"\n  Dataset size: {len(dataset)}")
        print(f"  Class weights: {self.class_weights}")
        
        return DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device == "cuda" else False,
            drop_last=True
        )
    
    def train_step(self, batch: Dict) -> Dict:
        """Execute single training step"""
        self.model.train()
        self.projection_image.train()
        self.projection_text.train()
        
        # Move to device
        images = batch['image'].to(self.device)
        texts = batch['text'].to(self.device)
        scene_classes = batch['scene_class']
        
        # Compute sample weights for rare class upweighting
        sample_weights = torch.tensor([
            self.class_weights.get(cls, 1.0) for cls in scene_classes
        ], device=self.device)
        
        # Forward pass through CLIP
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Project through heads
        image_projected = self.projection_image(image_features)
        text_projected = self.projection_text(text_features)
        
        # Compute losses
        clip_loss_val, clip_metrics = self.clip_loss(
            image_projected, text_projected, sample_weights
        )
        
        total_loss = clip_loss_val
        all_metrics = clip_metrics.copy()
        
        # Fine-grained loss (optional)
        if self.fine_grained_loss is not None:
            # For fine-grained, we need word-level features
            # Using full features as proxy
            fg_loss, fg_metrics = self.fine_grained_loss(
                text_features.unsqueeze(1),  # (B, 1, D) as word features
                image_features.unsqueeze(1)  # (B, 1, D) as region features
            )
            total_loss = total_loss + self.config.FINE_GRAINED_WEIGHT * fg_loss
            all_metrics.update(fg_metrics)
            all_metrics['fine_grained_loss'] = fg_loss.item()
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.projection_image.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.projection_text.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        all_metrics['total_loss'] = total_loss.item()
        all_metrics['lr'] = self.scheduler.get_last_lr()[0]
        
        return all_metrics
    
    def save_checkpoint(self, tag: str = "latest"):
        """Save model checkpoint"""
        checkpoint_dir = self.config.OUTPUT_DIR
        
        # Save LoRA weights
        lora_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lora_state[name] = param.data.cpu()
        
        torch.save(lora_state, checkpoint_dir / f"lora_{tag}.pt")
        
        # Save projection heads
        torch.save({
            'projection_image': self.projection_image.state_dict(),
            'projection_text': self.projection_text.state_dict()
        }, checkpoint_dir / f"projection_heads_{tag}.pt")
        
        # Save optimizer state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss
        }, checkpoint_dir / f"training_state_{tag}.pt")
        
        # Save training log
        with open(checkpoint_dir / "training_log.json", 'w') as f:
            json.dump({
                'config': {
                    'model_type': self.model_type,
                    'lora_rank': self.config.LORA_RANK,
                    'lora_alpha': self.config.LORA_ALPHA,
                    'synthetic_ratio': self.config.SYNTHETIC_RATIO,
                    'learning_rate': self.config.LEARNING_RATE,
                    'use_fine_grained_loss': self.config.USE_FINE_GRAINED_LOSS
                },
                'training_log': self.training_log
            }, f, indent=2)
        
        print(f"  ✓ Checkpoint saved: {tag}")
    
    def train(self):
        """Run full training loop"""
        print("\n" + "="*80)
        print("STAGE 2 TRAINING: Synthetic-Heavy LoRA Fine-Tuning")
        print("="*80)
        
        # Setup
        self.setup()
        
        # Create data loader
        print("\nCreating data loader...")
        dataloader = self.create_dataloader()
        
        print(f"\nStarting training for {self.config.NUM_EPOCHS} epochs")
        print(f"  Batch size: {self.config.BATCH_SIZE}")
        print(f"  Synthetic ratio: {self.config.SYNTHETIC_RATIO}")
        print(f"  Fine-grained loss: {'Enabled' if self.config.USE_FINE_GRAINED_LOSS else 'Disabled'}")
        
        try:
            for epoch in range(self.config.NUM_EPOCHS):
                epoch_losses = []
                
                print(f"\n--- Epoch {epoch + 1}/{self.config.NUM_EPOCHS} ---")
                
                for batch_idx, batch in enumerate(dataloader):
                    self.global_step += 1
                    
                    # Train step
                    metrics = self.train_step(batch)
                    epoch_losses.append(metrics['total_loss'])
                    
                    # Log
                    if self.global_step % self.config.LOG_INTERVAL == 0:
                        avg_loss = np.mean(epoch_losses[-self.config.LOG_INTERVAL:])
                        print(f"  Step {self.global_step}: loss={avg_loss:.4f}, "
                              f"i2t_acc={metrics['acc_i2t']:.3f}, "
                              f"t2i_acc={metrics['acc_t2i']:.3f}, "
                              f"lr={metrics['lr']:.6f}")
                        
                        self.training_log.append({
                            'step': self.global_step,
                            'epoch': epoch + 1,
                            **metrics
                        })
                    
                    # Save checkpoint
                    if self.global_step % self.config.SAVE_INTERVAL == 0:
                        self.save_checkpoint("latest")
                        
                        # Save best model
                        if metrics['total_loss'] < self.best_loss:
                            self.best_loss = metrics['total_loss']
                            self.save_checkpoint("best")
                
                # End of epoch summary
                avg_epoch_loss = np.mean(epoch_losses)
                print(f"\n  Epoch {epoch + 1} complete: avg_loss={avg_epoch_loss:.4f}")
                self.save_checkpoint(f"epoch_{epoch + 1}")
        
        except KeyboardInterrupt:
            print("\n\n⚠ Training interrupted! Saving checkpoint...")
            self.save_checkpoint("interrupted")
        
        # Final save
        self.save_checkpoint("final")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"  Final loss: {self.best_loss:.4f}")
        print(f"  Checkpoints saved to: {self.config.OUTPUT_DIR}")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Main entry point for Stage 2 training"""
    print("\n" + "="*80)
    print("STAGE 2: SYNTHETIC-HEAVY LoRA FINE-TUNING")
    print("="*80)
    
    if not HAS_TORCH:
        print("✗ PyTorch not available. Stage 2 requires GPU.")
        sys.exit(1)
    
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check for dataset
    config = Stage2Config()
    if not config.FILTERED_DATASET.exists():
        print(f"\n✗ Filtered dataset not found: {config.FILTERED_DATASET}")
        print("  Please run stage1_dataset_builder.py and quality_filter.py first.")
        sys.exit(1)
    
    # Run training
    trainer = Stage2Trainer(config)
    trainer.train()
    
    print("\n" + "="*80)
    print("STAGE 2 COMPLETE")
    print("="*80)
    print(f"Checkpoints: {config.OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Evaluate on test set using the trained LoRA adapters")
    print("  2. Export combined model for inference")


if __name__ == "__main__":
    main()
