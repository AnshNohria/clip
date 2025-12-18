#!/usr/bin/env python3
# type: ignore
"""
Stage 2: Synthetic-Heavy Fine-Tuning Trainer

Training with 80% high-quality synthetic pairs and 20% zoom crops.
- RemoteCLIP-ViT-B-32 with LoRA adapters (rank=16, alpha=32)
- Batch size 256 (effective 1024 via 4x gradient accumulation)
- BF16 mixed precision
- CLIP contrastive loss (temperature=0.07)
- Learning rate 5e-4 with cosine decay and 1k warmup
- 10-15 epochs
"""
from __future__ import annotations

import os
import json
import math
import time
import random
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Iterator, TYPE_CHECKING
import warnings
warnings.filterwarnings('ignore')

if TYPE_CHECKING:
    import torch as torch_t
    from torch.utils.data import DataLoader as DataLoaderType
    from PIL import Image as PILImage

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc,assignment]
    DataLoader = None  # type: ignore[misc,assignment]
    WeightedRandomSampler = None  # type: ignore[misc,assignment]

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None  # type: ignore[assignment,misc]
    np = None  # type: ignore[assignment]

from .config import PipelineConfig, TrainingConfig
from .fourier_augment import FourierAmplitudeSwap


# ============================================================================
# LoRA IMPLEMENTATION  
# ============================================================================

if HAS_TORCH:
    _nn_Module = nn.Module  # type: ignore[union-attr]
else:
    _nn_Module = object


class LoRALayer(_nn_Module):  # type: ignore[misc,valid-type]
    """Low-Rank Adaptation layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: Any, original_output: Any) -> Any:
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return original_output + lora_out * self.scaling


class LoRALinear(nn.Module):
    """Wrapper adding LoRA to a Linear layer."""
    
    def __init__(
        self,
        original_layer: Any,
        rank: int = 16,
        alpha: float = 32.0,
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
        
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: Any) -> Any:
        return self.lora(x, self.original_layer(x))


def apply_lora_to_model(
    model: Any,
    target_modules: List[str],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.1
) -> Any:
    """Apply LoRA to specified modules."""
    lora_count = 0
    
    for name, module in list(model.named_modules()):
        should_apply = any(target in name for target in target_modules)
        
        if should_apply and isinstance(module, nn.Linear):
            lora_linear = LoRALinear(
                original_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
            
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            setattr(parent, parts[-1], lora_linear)
            lora_count += 1
    
    print(f"  Applied LoRA to {lora_count} modules")
    return model


# ============================================================================
# DATASET
# ============================================================================

class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2 training.
    
    Combines synthetic samples and zoom crops with weighted sampling.
    """
    
    def __init__(
        self,
        synthetic_dataset_path: Path,
        crops_dataset_path: Path,
        transform: Any,
        tokenizer: Any,
        synthetic_ratio: float = 0.8,
        fourier_augment: Optional[FourierAmplitudeSwap] = None
    ):
        # Load datasets
        self.synthetic_pairs = self._load_dataset(synthetic_dataset_path)
        self.crop_pairs = self._load_dataset(crops_dataset_path)
        
        self.transform = transform
        self.tokenizer = tokenizer
        self.synthetic_ratio = synthetic_ratio
        self.fourier_augment = fourier_augment
        
        # Build combined list with source tracking
        self.all_pairs = []
        self.sample_weights = []
        
        # Add synthetic samples
        for pair in self.synthetic_pairs:
            self.all_pairs.append(('synthetic', pair))
            self.sample_weights.append(synthetic_ratio)
        
        # Add crop samples
        crop_weight = 1.0 - synthetic_ratio
        for pair in self.crop_pairs:
            self.all_pairs.append(('crop', pair))
            self.sample_weights.append(crop_weight)
        
        # Normalize weights
        total_weight = sum(self.sample_weights)
        self.sample_weights = [w / total_weight for w in self.sample_weights]
        
        print(f"  Loaded {len(self.synthetic_pairs)} synthetic, {len(self.crop_pairs)} crops")
        print(f"  Total: {len(self.all_pairs)} pairs")
    
    def _load_dataset(self, path: Path) -> List[Dict]:
        """Load dataset from JSON."""
        if not path.exists():
            print(f"  ⚠ Dataset not found: {path}")
            return []
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data.get('pairs', [])
    
    def __len__(self) -> int:
        return len(self.all_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        source, pair = self.all_pairs[idx]
        
        # Load image
        image_path = pair.get('image_path', '')
        if not os.path.isabs(image_path):
            # Assume relative to output dir
            image_path = str(Path("outputs/remoteclip_pipeline") / image_path)
        
        try:
            if Image is None:
                raise RuntimeError("PIL required")
            image = Image.open(image_path).convert('RGB')
            
            # Apply Fourier augmentation to crops
            if source == 'crop' and self.fourier_augment is not None:
                if random.random() < 0.3:  # 30% probability
                    # Get another random crop for amplitude swap
                    other_idx = random.randint(0, len(self.crop_pairs) - 1)
                    other_pair = self.crop_pairs[other_idx]
                    other_path = other_pair.get('image_path', '')
                    if not os.path.isabs(other_path):
                        other_path = str(Path("outputs/remoteclip_pipeline") / other_path)
                    try:
                        other_image = Image.open(other_path).convert('RGB')
                        image = self.fourier_augment(image, other_image)
                    except:
                        pass
            
            image = self.transform(image)
            
        except Exception as e:
            # Dummy image on error
            image = torch.zeros(3, 224, 224)
        
        # Tokenize caption
        caption = pair.get('caption', pair.get('refined_caption', ''))
        text_tokens = self.tokenizer([caption])[0]
        
        return {
            'image': image,
            'text': text_tokens,
            'caption': caption,
            'source': source,
            'image_id': pair.get('image_id', pair.get('crop_id', str(idx)))
        }
    
    def get_sampler(self) -> WeightedRandomSampler:
        """Get weighted sampler for 80/20 ratio per batch."""
        weights = torch.tensor(self.sample_weights, dtype=torch.float)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.all_pairs),
            replacement=True
        )


# ============================================================================
# LOSSES
# ============================================================================

class CLIPContrastiveLoss(nn.Module):
    """CLIP contrastive loss (InfoNCE)."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1/temperature)))
    
    def forward(
        self,
        image_features: Any,
        text_features: Any,
        sample_weights: Optional[Any] = None
    ) -> Tuple[Any, Dict]:
        temperature = torch.exp(self.log_temperature).clamp(min=0.01, max=100)
        
        logits = (image_features @ text_features.T) / temperature
        batch_size = image_features.shape[0]
        targets = torch.arange(batch_size, device=logits.device)
        
        loss_i2t = F.cross_entropy(logits, targets, reduction='none')
        loss_t2i = F.cross_entropy(logits.T, targets, reduction='none')
        
        if sample_weights is not None:
            loss_i2t = loss_i2t * sample_weights
            loss_t2i = loss_t2i * sample_weights
        
        loss = (loss_i2t.mean() + loss_t2i.mean()) / 2
        
        with torch.no_grad():
            i2t_acc = (logits.argmax(dim=1) == targets).float().mean()
            t2i_acc = (logits.argmax(dim=0) == targets).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'loss_i2t': loss_i2t.mean().item(),
            'loss_t2i': loss_t2i.mean().item(),
            'acc_i2t': i2t_acc.item(),
            'acc_t2i': t2i_acc.item(),
            'temperature': temperature.item()
        }
        
        return loss, metrics


# ============================================================================
# TRAINER
# ============================================================================

class Stage2SyntheticHeavyTrainer:
    """
    Stage 2 Trainer: Synthetic-Heavy Fine-Tuning.
    
    80% synthetic + 20% zoom crops with LoRA adapters.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.train_config = config.training
        self.device = config.device
        
        # Models
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.preprocess: Optional[Any] = None
        self.model_type: Optional[str] = None
        
        # Training components
        self.optimizer: Optional[Any] = None
        self.scheduler: Optional[Any] = None
        self.clip_loss: Optional[Any] = None
        self.scaler: Optional[Any] = None
        
        # State
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_log: List[Dict] = []
        
        # Paths
        self.checkpoint_dir = config.checkpoint_dir / "stage2"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Background checkpointing
        self.checkpoint_thread: Optional[threading.Thread] = None
    
    def setup(self):
        """Initialize all components."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for Stage 2 training")
        
        print("\n" + "="*80)
        print("STAGE 2 SETUP: Synthetic-Heavy Training")
        print("="*80)
        
        # Load CLIP model
        print("\n[1/4] Loading RemoteCLIP backbone...")
        self._load_clip_model()
        
        # Apply LoRA
        print("\n[2/4] Applying LoRA adapters...")
        self._apply_lora()
        
        # Setup optimizer
        print("\n[3/4] Setting up optimizer...")
        self._setup_optimizer()
        
        # Setup loss
        print("\n[4/4] Setting up loss functions...")
        self._setup_losses()
        
        print("\n✓ Stage 2 setup complete")
    
    def _load_clip_model(self):
        """Load RemoteCLIP or fallback CLIP."""
        try:
            import open_clip
            
            # Try RemoteCLIP
            checkpoint_dir = self.train_config.remoteclip_checkpoint
            checkpoints = list(checkpoint_dir.rglob("*.pt")) if checkpoint_dir.exists() else []
            checkpoints.extend(checkpoint_dir.rglob("*.bin") if checkpoint_dir.exists() else [])
            
            if checkpoints:
                print(f"  Loading RemoteCLIP from {checkpoints[0]}")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained=str(checkpoints[0])
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
                        'ViT-B-32', pretrained=checkpoint_path
                    )
                    self.model_type = "RemoteCLIP"
                    print("  ✓ Loaded RemoteCLIP from HuggingFace")
                except:
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        'ViT-B-32', pretrained='openai'
                    )
                    self.model_type = "OpenCLIP"
                    print("  ✓ Loaded OpenCLIP (fallback)")
            
            self.model = model.to(self.device)
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            # Freeze backbone
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(f"  Model: {self.model_type}")
            print(f"  Device: {self.device}")
            
        except ImportError:
            raise RuntimeError("open_clip required for Stage 2 training")
    
    def _apply_lora(self):
        """Apply LoRA to Q/V projections."""
        assert self.model is not None
        
        # Apply to visual encoder
        if hasattr(self.model, 'visual'):
            self.model.visual = apply_lora_to_model(
                self.model.visual,
                target_modules=self.train_config.lora_target_modules,
                rank=self.train_config.lora_rank,
                alpha=self.train_config.lora_alpha,
                dropout=self.train_config.lora_dropout
            )
        
        # Apply to text encoder
        if hasattr(self.model, 'transformer'):
            self.model.transformer = apply_lora_to_model(
                self.model.transformer,
                target_modules=self.train_config.lora_target_modules,
                rank=self.train_config.lora_rank,
                alpha=self.train_config.lora_alpha,
                dropout=self.train_config.lora_dropout
            )
        
        # Count parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer with cosine scheduler."""
        assert self.model is not None
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.train_config.stage2_learning_rate,
            weight_decay=0.01
        )
        
        # Cosine scheduler with warmup
        def lr_lambda(step: int) -> float:
            if step < self.train_config.stage2_warmup_steps:
                return step / self.train_config.stage2_warmup_steps
            total_steps = self.train_config.stage2_epochs * 1000
            progress = (step - self.train_config.stage2_warmup_steps) / \
                      (total_steps - self.train_config.stage2_warmup_steps + 1)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision scaler
        if self.train_config.use_bf16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        print(f"  LR: {self.train_config.stage2_learning_rate}")
        print(f"  Warmup: {self.train_config.stage2_warmup_steps} steps")
        print(f"  BF16: {self.train_config.use_bf16}")
    
    def _setup_losses(self):
        """Setup loss functions."""
        self.clip_loss = CLIPContrastiveLoss(
            temperature=self.train_config.temperature
        ).to(self.device)
        print(f"  Temperature: {self.train_config.temperature}")
    
    def create_dataloader(
        self,
        synthetic_path: Path,
        crops_path: Path
    ) -> Any:
        """Create training Any."""
        assert self.preprocess is not None and self.tokenizer is not None
        
        # Fourier augmentation
        fourier_aug = FourierAmplitudeSwap(
            swap_ratio=self.train_config.fourier_swap_ratio
        )
        
        dataset = Stage2Dataset(
            synthetic_dataset_path=synthetic_path,
            crops_dataset_path=crops_path,
            transform=self.preprocess,
            tokenizer=self.tokenizer,
            synthetic_ratio=self.train_config.stage2_synthetic_ratio,
            fourier_augment=fourier_aug
        )
        
        sampler = dataset.get_sampler()
        
        return DataLoader(
            dataset,
            batch_size=self.train_config.stage2_batch_size,
            sampler=sampler,
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory,
            prefetch_factor=self.train_config.prefetch_factor,
            persistent_workers=True,
            drop_last=True
        )
    
    def train_step(self, batch: Dict) -> Dict:
        """Execute single training step with gradient accumulation."""
        assert self.model is not None
        assert self.optimizer is not None
        assert self.clip_loss is not None
        
        self.model.train()
        
        images = batch['image'].to(self.device)
        texts = batch['text'].to(self.device)
        
        # Mixed precision forward
        with torch.cuda.amp.autocast(enabled=self.train_config.use_bf16, dtype=torch.bfloat16):
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)
            
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            loss, metrics = self.clip_loss(image_features, text_features)
            loss = loss / self.train_config.stage2_grad_accum_steps
        
        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return metrics
    
    def optimizer_step(self):
        """Execute optimizer step after gradient accumulation."""
        assert self.optimizer is not None
        assert self.scheduler is not None
        
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    def save_checkpoint(self, tag: str = "latest", async_save: bool = True):
        """Save checkpoint (optionally async)."""
        if async_save and self.train_config.async_checkpoint:
            # Wait for previous checkpoint to finish
            if self.checkpoint_thread is not None:
                self.checkpoint_thread.join()
            
            self.checkpoint_thread = threading.Thread(
                target=self._save_checkpoint_sync,
                args=(tag,)
            )
            self.checkpoint_thread.start()
        else:
            self._save_checkpoint_sync(tag)
    
    def _save_checkpoint_sync(self, tag: str):
        """Synchronous checkpoint save."""
        assert self.model is not None
        assert self.optimizer is not None
        assert self.scheduler is not None
        
        # Save LoRA weights
        lora_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lora_state[name] = param.data.cpu()
        
        torch.save(lora_state, self.checkpoint_dir / f"lora_{tag}.pt")
        
        # Save training state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'scaler': self.scaler.state_dict() if self.scaler else None
        }, self.checkpoint_dir / f"training_state_{tag}.pt")
        
        # Save log
        with open(self.checkpoint_dir / "training_log.json", 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'config': {
                    'lora_rank': self.train_config.lora_rank,
                    'lora_alpha': self.train_config.lora_alpha,
                    'synthetic_ratio': self.train_config.stage2_synthetic_ratio,
                    'learning_rate': self.train_config.stage2_learning_rate,
                },
                'log': self.training_log
            }, f, indent=2)
    
    def train(
        self,
        synthetic_dataset_path: Path,
        crops_dataset_path: Path
    ) -> Dict[str, Any]:
        """Run full Stage 2 training."""
        print("\n" + "="*80)
        print("STAGE 2: SYNTHETIC-HEAVY FINE-TUNING")
        print("="*80)
        
        # Setup
        self.setup()
        
        # Create Any
        print("\nCreating Any...")
        Any = self.create_dataloader(synthetic_dataset_path, crops_dataset_path)
        
        print(f"\nTraining configuration:")
        print(f"  Epochs: {self.train_config.stage2_epochs}")
        print(f"  Batch size: {self.train_config.stage2_batch_size}")
        print(f"  Effective batch: {self.train_config.stage2_batch_size * self.train_config.stage2_grad_accum_steps}")
        print(f"  Synthetic ratio: {self.train_config.stage2_synthetic_ratio}")
        print(f"  Gradient accumulation: {self.train_config.stage2_grad_accum_steps}")
        
        start_time = time.time()
        accum_metrics: Dict[str, float] = {}
        
        for epoch in range(self.train_config.stage2_epochs):
            epoch_losses = []
            print(f"\n--- Epoch {epoch + 1}/{self.train_config.stage2_epochs} ---")
            
            for batch_idx, batch in enumerate(Any):
                self.global_step += 1
                
                # Training step
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                
                # Accumulate metrics
                for k, v in metrics.items():
                    accum_metrics[k] = accum_metrics.get(k, 0) + v
                
                # Optimizer step after accumulation
                if self.global_step % self.train_config.stage2_grad_accum_steps == 0:
                    self.optimizer_step()
                    
                    # Log
                    if self.global_step % self.config.log_interval == 0:
                        avg_metrics = {k: v / self.train_config.stage2_grad_accum_steps 
                                      for k, v in accum_metrics.items()}
                        
                        assert self.scheduler is not None
                        print(f"  Step {self.global_step}: "
                              f"loss={avg_metrics['loss']:.4f}, "
                              f"acc_i2t={avg_metrics['acc_i2t']:.3f}, "
                              f"lr={self.scheduler.get_last_lr()[0]:.2e}")
                        
                        self.training_log.append({
                            'step': self.global_step,
                            'epoch': epoch,
                            **avg_metrics
                        })
                        
                        accum_metrics = {}
                    
                    # Checkpoint
                    if self.global_step % self.train_config.checkpoint_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
            
            # Epoch summary
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"  Epoch {epoch + 1} loss: {epoch_loss:.4f}")
            
            # Save best
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint("best", async_save=False)
        
        # Final save
        self.save_checkpoint("final", async_save=False)
        
        # Wait for any async checkpoints
        if self.checkpoint_thread is not None:
            self.checkpoint_thread.join()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("STAGE 2 TRAINING COMPLETE")
        print("="*80)
        print(f"  Total time: {elapsed/3600:.2f}h")
        print(f"  Best loss: {self.best_loss:.4f}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        
        return {
            'best_loss': self.best_loss,
            'total_steps': self.global_step,
            'elapsed_time': elapsed,
            'checkpoint_dir': str(self.checkpoint_dir)
        }
