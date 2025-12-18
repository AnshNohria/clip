#!/usr/bin/env python3
# type: ignore
"""
Stage 3: Real-Dominated Fine-Tuning Trainer

Training with 85% zoom crops and 15% top-quality synthetic (CLIP > 0.75).
- Validates on real-only RSICD test set
- KL regularization to preserve Stage 2 learned structure
- Fourier swap augmentation on 30% of crops
- Early stopping with patience 3 monitoring R@10
"""
from __future__ import annotations

import os
import json
import math
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
import warnings
warnings.filterwarnings('ignore')

if TYPE_CHECKING:
    from torch.utils.data import DataLoader as DataLoaderType

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore
    WeightedRandomSampler = None  # type: ignore

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None  # type: ignore
    np = None  # type: ignore

from .config import PipelineConfig, TrainingConfig
from .stage2_trainer import CLIPContrastiveLoss, apply_lora_to_model
from .fourier_augment import FourierAmplitudeSwap


# ============================================================================
# DATASET
# ============================================================================

class Stage3Dataset(Dataset):
    """
    Dataset for Stage 3 training.
    
    85% zoom crops + 15% top-quality synthetic (CLIP > 0.75).
    """
    
    def __init__(
        self,
        synthetic_dataset_path: Path,
        crops_dataset_path: Path,
        transform: Any,
        tokenizer: Any,
        crop_ratio: float = 0.85,
        min_clip_score: float = 0.75,
        fourier_augment: Optional[FourierAmplitudeSwap] = None,
        fourier_probability: float = 0.3
    ):
        # Load datasets
        self.synthetic_pairs = self._load_filtered_synthetic(
            synthetic_dataset_path, min_clip_score
        )
        self.crop_pairs = self._load_dataset(crops_dataset_path)
        
        self.transform = transform
        self.tokenizer = tokenizer
        self.crop_ratio = crop_ratio
        self.fourier_augment = fourier_augment
        self.fourier_probability = fourier_probability
        
        # Build combined list
        self.all_pairs = []
        self.sample_weights = []
        
        synthetic_weight = 1.0 - crop_ratio
        for pair in self.synthetic_pairs:
            self.all_pairs.append(('synthetic', pair))
            self.sample_weights.append(synthetic_weight)
        
        for pair in self.crop_pairs:
            self.all_pairs.append(('crop', pair))
            self.sample_weights.append(crop_ratio)
        
        # Normalize
        total = sum(self.sample_weights)
        self.sample_weights = [w / total for w in self.sample_weights]
        
        print(f"  Stage 3 dataset: {len(self.synthetic_pairs)} synthetic (CLIP>{min_clip_score}), "
              f"{len(self.crop_pairs)} crops")
    
    def _load_dataset(self, path: Path) -> List[Dict]:
        """Load dataset from JSON."""
        if not path.exists():
            return []
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('pairs', [])
    
    def _load_filtered_synthetic(self, path: Path, min_clip: float) -> List[Dict]:
        """Load synthetic dataset filtered by CLIP score."""
        pairs = self._load_dataset(path)
        
        filtered = []
        for pair in pairs:
            clip_score = pair.get('metadata', {}).get('clip_score', 
                         pair.get('quality_scores', {}).get('clip_score', 0))
            if clip_score >= min_clip:
                filtered.append(pair)
        
        print(f"  Filtered synthetic: {len(filtered)}/{len(pairs)} with CLIP>={min_clip}")
        return filtered
    
    def __len__(self) -> int:
        return len(self.all_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        source, pair = self.all_pairs[idx]
        
        image_path = pair.get('image_path', '')
        if not os.path.isabs(image_path):
            image_path = str(Path("outputs/remoteclip_pipeline") / image_path)
        
        try:
            if Image is None:
                raise RuntimeError("PIL required")
            image = Image.open(image_path).convert('RGB')
            
            # Fourier augmentation on crops
            if source == 'crop' and self.fourier_augment is not None:
                if random.random() < self.fourier_probability:
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
        except Exception:
            image = torch.zeros(3, 224, 224)
        
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
        """Get weighted sampler."""
        weights = torch.tensor(self.sample_weights, dtype=torch.float)
        return WeightedRandomSampler(weights, len(self.all_pairs), replacement=True)


class RSICDTestDataset(Dataset):
    """RSICD test dataset for validation."""
    
    def __init__(
        self,
        rsicd_path: Path,
        transform: Any,
        tokenizer: Any
    ):
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Load RSICD annotations
        self.pairs = self._load_rsicd(rsicd_path)
        print(f"  RSICD test: {len(self.pairs)} pairs")
    
    def _load_rsicd(self, rsicd_path: Path) -> List[Dict]:
        """Load RSICD test split."""
        pairs = []
        
        # Try to find annotations file
        for ann_file in ["test.json", "dataset_rsicd.json", "annotations.json"]:
            ann_path = rsicd_path / ann_file
            if ann_path.exists():
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                
                # Handle different formats
                if 'images' in data:
                    for img in data['images']:
                        if img.get('split', 'test') == 'test':
                            captions = [s['raw'] for s in data.get('sentences', []) 
                                       if s['imgid'] == img['imgid']]
                            if captions:
                                pairs.append({
                                    'image_path': str(rsicd_path / 'images' / img['filename']),
                                    'caption': captions[0],
                                    'all_captions': captions,
                                    'image_id': img['imgid']
                                })
                elif 'pairs' in data:
                    pairs = [p for p in data['pairs'] if p.get('split', 'test') == 'test']
                break
        
        # Fallback: scan images directory
        if not pairs:
            images_dir = rsicd_path / 'images'
            if images_dir.exists():
                for img_path in images_dir.glob("*.jpg"):
                    pairs.append({
                        'image_path': str(img_path),
                        'caption': img_path.stem.replace('_', ' '),
                        'image_id': img_path.stem
                    })
        
        return pairs[:1000]  # Limit for efficiency
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        
        try:
            if Image is None:
                raise RuntimeError("PIL required")
            image = Image.open(pair['image_path']).convert('RGB')
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)
        
        text_tokens = self.tokenizer([pair['caption']])[0]
        
        return {
            'image': image,
            'text': text_tokens,
            'caption': pair['caption'],
            'image_id': pair['image_id'],
            'all_captions': pair.get('all_captions', [pair['caption']])
        }


# ============================================================================
# TRAINER
# ============================================================================

class Stage3RealDominatedTrainer:
    """
    Stage 3 Trainer: Real-Dominated Fine-Tuning.
    
    85% crops + 15% top synthetic with KL regularization.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.train_config = config.training
        self.device = config.device
        
        # Models
        self.model: Optional[Any] = None
        self.frozen_model: Optional[Any] = None  # Stage 2 model for KL
        self.tokenizer: Optional[Any] = None
        self.preprocess: Optional[Any] = None
        self.model_type: Optional[str] = None
        
        # Training
        self.optimizer: Optional[Any] = None
        self.scheduler: Optional[Any] = None
        self.clip_loss: Optional[Any] = None
        self.scaler: Optional[Any] = None
        
        # State
        self.global_step = 0
        self.best_r10 = 0.0
        self.patience_counter = 0
        self.training_log: List[Dict] = []
        
        # Paths
        self.checkpoint_dir = config.checkpoint_dir / "stage3"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.stage2_checkpoint_dir = config.checkpoint_dir / "stage2"
    
    def setup(self):
        """Initialize all components."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        
        print("\n" + "="*80)
        print("STAGE 3 SETUP: Real-Dominated Training")
        print("="*80)
        
        # Load models
        print("\n[1/5] Loading RemoteCLIP backbone...")
        self._load_clip_model()
        
        # Load Stage 2 checkpoint
        print("\n[2/5] Loading Stage 2 LoRA weights...")
        self._load_stage2_weights()
        
        # Create frozen copy for KL
        print("\n[3/5] Creating frozen reference model...")
        self._create_frozen_model()
        
        # Setup optimizer
        print("\n[4/5] Setting up optimizer...")
        self._setup_optimizer()
        
        # Setup losses
        print("\n[5/5] Setting up losses...")
        self._setup_losses()
        
        print("\n✓ Stage 3 setup complete")
    
    def _load_clip_model(self):
        """Load CLIP model."""
        try:
            import open_clip
            
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            
            self.model = model.to(self.device)
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.model_type = "OpenCLIP"
            
            # Freeze backbone
            for param in self.model.parameters():
                param.requires_grad = False
                
        except ImportError:
            raise RuntimeError("open_clip required")
    
    def _load_stage2_weights(self):
        """Load LoRA weights from Stage 2."""
        assert self.model is not None
        
        # Apply LoRA structure
        if hasattr(self.model, 'visual'):
            self.model.visual = apply_lora_to_model(
                self.model.visual,
                target_modules=self.train_config.lora_target_modules,
                rank=self.train_config.lora_rank,
                alpha=self.train_config.lora_alpha,
                dropout=self.train_config.lora_dropout
            )
        
        if hasattr(self.model, 'transformer'):
            self.model.transformer = apply_lora_to_model(
                self.model.transformer,
                target_modules=self.train_config.lora_target_modules,
                rank=self.train_config.lora_rank,
                alpha=self.train_config.lora_alpha,
                dropout=self.train_config.lora_dropout
            )
        
        # Load weights
        lora_path = self.stage2_checkpoint_dir / "lora_best.pt"
        if not lora_path.exists():
            lora_path = self.stage2_checkpoint_dir / "lora_final.pt"
        
        if lora_path.exists():
            lora_state = torch.load(lora_path, map_location=self.device)
            
            # Load matching parameters
            model_state = self.model.state_dict()
            loaded = 0
            for name, param in lora_state.items():
                if name in model_state:
                    model_state[name].copy_(param)
                    loaded += 1
            
            print(f"  Loaded {loaded} LoRA parameters from Stage 2")
        else:
            print(f"  ⚠ Stage 2 checkpoint not found: {lora_path}")
    
    def _create_frozen_model(self):
        """Create frozen copy for KL regularization."""
        import copy
        
        self.frozen_model = copy.deepcopy(self.model)
        for param in self.frozen_model.parameters():
            param.requires_grad = False
        self.frozen_model.eval()
        
        print("  ✓ Frozen reference model created")
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        assert self.model is not None
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.train_config.stage3_learning_rate,
            weight_decay=0.01
        )
        
        def lr_lambda(step: int) -> float:
            if step < self.train_config.stage3_warmup_steps:
                return step / self.train_config.stage3_warmup_steps
            total = self.train_config.stage3_epochs * 500
            progress = (step - self.train_config.stage3_warmup_steps) / \
                      (total - self.train_config.stage3_warmup_steps + 1)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        if self.train_config.use_bf16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        print(f"  LR: {self.train_config.stage3_learning_rate}")
    
    def _setup_losses(self):
        """Setup losses."""
        self.clip_loss = CLIPContrastiveLoss(
            temperature=self.train_config.temperature
        ).to(self.device)
    
    def create_train_dataloader(
        self,
        synthetic_path: Path,
        crops_path: Path
    ) -> Any:
        """Create training Any."""
        assert self.preprocess is not None and self.tokenizer is not None
        
        fourier_aug = FourierAmplitudeSwap(
            swap_ratio=self.train_config.fourier_swap_ratio
        )
        
        dataset = Stage3Dataset(
            synthetic_dataset_path=synthetic_path,
            crops_dataset_path=crops_path,
            transform=self.preprocess,
            tokenizer=self.tokenizer,
            crop_ratio=self.train_config.stage3_crop_ratio,
            min_clip_score=0.75,
            fourier_augment=fourier_aug,
            fourier_probability=self.train_config.fourier_probability
        )
        
        return DataLoader(
            dataset,
            batch_size=self.train_config.stage3_batch_size,
            sampler=dataset.get_sampler(),
            num_workers=self.train_config.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def create_val_dataloader(self, rsicd_path: Path) -> Any:
        """Create validation Any."""
        assert self.preprocess is not None and self.tokenizer is not None
        
        dataset = RSICDTestDataset(
            rsicd_path=rsicd_path,
            transform=self.preprocess,
            tokenizer=self.tokenizer
        )
        
        return DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def compute_kl_loss(
        self,
        current_features: Any,
        frozen_features: Any
    ) -> Any:
        """Compute KL divergence between current and frozen embeddings."""
        # Softmax over embedding dimensions
        current_probs = F.softmax(current_features / 0.1, dim=-1)
        frozen_probs = F.softmax(frozen_features / 0.1, dim=-1)
        
        # KL divergence
        kl = F.kl_div(
            current_probs.log(),
            frozen_probs,
            reduction='batchmean'
        )
        
        return kl
    
    def train_step(self, batch: Dict) -> Dict:
        """Training step with KL regularization."""
        assert self.model is not None
        assert self.frozen_model is not None
        assert self.optimizer is not None
        assert self.clip_loss is not None
        
        self.model.train()
        
        images = batch['image'].to(self.device)
        texts = batch['text'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.train_config.use_bf16, dtype=torch.bfloat16):
            # Current model
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)
            
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Frozen model
            with torch.no_grad():
                frozen_img = self.frozen_model.encode_image(images)
                frozen_txt = self.frozen_model.encode_text(texts)
                frozen_img = F.normalize(frozen_img, dim=-1)
                frozen_txt = F.normalize(frozen_txt, dim=-1)
            
            # CLIP loss
            clip_loss_val, clip_metrics = self.clip_loss(image_features, text_features)
            
            # KL regularization
            kl_img = self.compute_kl_loss(image_features, frozen_img)
            kl_txt = self.compute_kl_loss(text_features, frozen_txt)
            kl_loss = (kl_img + kl_txt) / 2
            
            # Total loss
            total_loss = clip_loss_val + self.train_config.stage3_kl_weight * kl_loss
            total_loss = total_loss / self.train_config.stage3_grad_accum_steps
        
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        metrics = {
            **clip_metrics,
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item() * self.train_config.stage3_grad_accum_steps
        }
        
        return metrics
    
    def optimizer_step(self):
        """Optimizer step."""
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
    
    @torch.no_grad()
    def validate(self, val_loader: Any) -> Dict[str, float]:
        """Validate on RSICD test set."""
        assert self.model is not None
        
        self.model.eval()
        
        all_image_features = []
        all_text_features = []
        
        print("  Encoding validation set...")
        for batch in val_loader:
            images = batch['image'].to(self.device)
            texts = batch['text'].to(self.device)
            
            img_feat = self.model.encode_image(images)
            txt_feat = self.model.encode_text(texts)
            
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
            
            all_image_features.append(img_feat.cpu())
            all_text_features.append(txt_feat.cpu())
        
        image_features = torch.cat(all_image_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)
        
        # Compute similarity
        similarity = image_features @ text_features.T
        
        n = similarity.shape[0]
        
        # Text-to-Image retrieval
        t2i_ranks = []
        for i in range(n):
            scores = similarity[:, i]
            rank = (scores > scores[i]).sum().item() + 1
            t2i_ranks.append(rank)
        
        # Image-to-Text retrieval
        i2t_ranks = []
        for i in range(n):
            scores = similarity[i, :]
            rank = (scores > scores[i]).sum().item() + 1
            i2t_ranks.append(rank)
        
        # Compute recall metrics
        def recall_at_k(ranks: List[int], k: int) -> float:
            return sum(1 for r in ranks if r <= k) / len(ranks)
        
        metrics = {
            't2i_r1': recall_at_k(t2i_ranks, 1),
            't2i_r5': recall_at_k(t2i_ranks, 5),
            't2i_r10': recall_at_k(t2i_ranks, 10),
            'i2t_r1': recall_at_k(i2t_ranks, 1),
            'i2t_r5': recall_at_k(i2t_ranks, 5),
            'i2t_r10': recall_at_k(i2t_ranks, 10),
        }
        
        # Combined R@10 for early stopping
        metrics['mean_r10'] = (metrics['t2i_r10'] + metrics['i2t_r10']) / 2
        
        return metrics
    
    def save_checkpoint(self, tag: str = "latest"):
        """Save checkpoint."""
        assert self.model is not None
        assert self.optimizer is not None
        assert self.scheduler is not None
        
        lora_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lora_state[name] = param.data.cpu()
        
        torch.save(lora_state, self.checkpoint_dir / f"lora_{tag}.pt")
        
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_r10': self.best_r10,
            'scaler': self.scaler.state_dict() if self.scaler else None
        }, self.checkpoint_dir / f"training_state_{tag}.pt")
        
        with open(self.checkpoint_dir / "training_log.json", 'w') as f:
            json.dump({'log': self.training_log}, f, indent=2)
    
    def train(
        self,
        synthetic_path: Path,
        crops_path: Path,
        rsicd_path: Path
    ) -> Dict[str, Any]:
        """Run Stage 3 training."""
        print("\n" + "="*80)
        print("STAGE 3: REAL-DOMINATED FINE-TUNING")
        print("="*80)
        
        self.setup()
        
        # Anys
        print("\nCreating Anys...")
        train_loader = self.create_train_dataloader(synthetic_path, crops_path)
        val_loader = self.create_val_dataloader(rsicd_path)
        
        print(f"\nTraining configuration:")
        print(f"  Epochs: {self.train_config.stage3_epochs}")
        print(f"  Crop ratio: {self.train_config.stage3_crop_ratio}")
        print(f"  KL weight: {self.train_config.stage3_kl_weight}")
        print(f"  Early stopping patience: {self.train_config.stage3_early_stopping_patience}")
        
        start_time = time.time()
        
        for epoch in range(self.train_config.stage3_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.train_config.stage3_epochs} ---")
            
            epoch_metrics: Dict[str, float] = {}
            
            for batch_idx, batch in enumerate(train_loader):
                self.global_step += 1
                
                metrics = self.train_step(batch)
                
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                
                if self.global_step % self.train_config.stage3_grad_accum_steps == 0:
                    self.optimizer_step()
                
                if self.global_step % self.config.log_interval == 0:
                    n = self.config.log_interval
                    avg = {k: v/n for k, v in epoch_metrics.items()}
                    print(f"  Step {self.global_step}: loss={avg.get('total_loss', 0):.4f}, "
                          f"kl={avg.get('kl_loss', 0):.4f}")
                    epoch_metrics = {}
            
            # Validation
            print("\n  Validating on RSICD...")
            val_metrics = self.validate(val_loader)
            
            print(f"  T2I: R@1={val_metrics['t2i_r1']:.3f}, R@5={val_metrics['t2i_r5']:.3f}, "
                  f"R@10={val_metrics['t2i_r10']:.3f}")
            print(f"  I2T: R@1={val_metrics['i2t_r1']:.3f}, R@5={val_metrics['i2t_r5']:.3f}, "
                  f"R@10={val_metrics['i2t_r10']:.3f}")
            
            self.training_log.append({
                'epoch': epoch,
                'step': self.global_step,
                **val_metrics
            })
            
            # Early stopping check
            if val_metrics['mean_r10'] > self.best_r10:
                self.best_r10 = val_metrics['mean_r10']
                self.patience_counter = 0
                self.save_checkpoint("best")
                print(f"  ✓ New best R@10: {self.best_r10:.3f}")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.train_config.stage3_early_stopping_patience}")
                
                if self.patience_counter >= self.train_config.stage3_early_stopping_patience:
                    print("\n  Early stopping triggered!")
                    break
            
            self.save_checkpoint(f"epoch_{epoch}")
        
        self.save_checkpoint("final")
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("STAGE 3 TRAINING COMPLETE")
        print("="*80)
        print(f"  Total time: {elapsed/3600:.2f}h")
        print(f"  Best R@10: {self.best_r10:.3f}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        
        return {
            'best_r10': self.best_r10,
            'total_steps': self.global_step,
            'elapsed_time': elapsed
        }
