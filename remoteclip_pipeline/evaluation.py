#!/usr/bin/env python3
# type: ignore
"""
Evaluation and Retrieval System

Comprehensive evaluation for RemoteCLIP:
- Real image index on GPU
- Text query retrieval with top-K
- Intra-class and inter-class similarity metrics
- Zero-shot classification
"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, Any
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .config import PipelineConfig, EvaluationConfig


# ============================================================================
# DATASETS
# ============================================================================

class ImageIndexDataset(Dataset):
    """Dataset for building image index."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[str]],
        transform: Any
    ):
        self.image_paths = image_paths
        self.labels = labels or [Path(p).stem for p in image_paths]
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        path = self.image_paths[idx]
        
        try:
            if Image is None:
                raise RuntimeError("PIL required")
            image = Image.open(path).convert('RGB')
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)
        
        return {
            'image': image,
            'path': path,
            'label': self.labels[idx]
        }


# ============================================================================
# RETRIEVAL INDEX
# ============================================================================

class ImageRetrievalIndex:
    """
    GPU-based image retrieval index.
    
    Stores image embeddings on GPU for fast similarity search.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.features: Optional[Any] = None
        self.paths: List[str] = []
        self.labels: List[str] = []
        self.label_to_indices: Dict[str, List[int]] = defaultdict(list)
    
    def build(
        self,
        model: Any,
        data_loader: Any,
        show_progress: bool = True
    ):
        """Build index from Any."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        
        model.eval()
        all_features = []
        
        if show_progress:
            print("  Building image index...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch['image'].to(self.device)
                
                features = model.encode_image(images)
                features = F.normalize(features, dim=-1)
                
                all_features.append(features)
                self.paths.extend(batch['path'])
                self.labels.extend(batch['label'])
                
                if show_progress and (batch_idx + 1) % 10 == 0:
                    print(f"    Encoded {len(self.paths)} images...")
        
        self.features = torch.cat(all_features, dim=0)
        
        # Build label index
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        
        if show_progress:
            print(f"  ✓ Index built: {self.features.shape[0]} images, "
                  f"{len(self.label_to_indices)} classes")
    
    def search(
        self,
        query_features: Any,
        top_k: int = 10
    ) -> Tuple[Any, Any]:
        """
        Search for similar images.
        
        Args:
            query_features: Query embeddings [N, D]
            top_k: Number of results per query
        
        Returns:
            (scores, indices) tensors
        """
        assert self.features is not None
        
        if query_features.device != self.features.device:
            query_features = query_features.to(self.features.device)
        
        # Compute similarities
        similarities = query_features @ self.features.T
        
        # Get top-k
        scores, indices = torch.topk(similarities, k=min(top_k, self.features.shape[0]), dim=-1)
        
        return scores, indices
    
    def get_paths(self, indices: Any) -> List[List[str]]:
        """Get paths for indices."""
        result = []
        indices_np = indices.cpu().numpy()
        
        for query_indices in indices_np:
            result.append([self.paths[i] for i in query_indices])
        
        return result
    
    def save(self, path: Path):
        """Save index to disk."""
        assert self.features is not None
        
        torch.save({
            'features': self.features.cpu(),
            'paths': self.paths,
            'labels': self.labels
        }, path)
    
    def load(self, path: Path):
        """Load index from disk."""
        data = torch.load(path, map_location=self.device)
        self.features = data['features'].to(self.device)
        self.paths = data['paths']
        self.labels = data['labels']
        
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)


# ============================================================================
# METRICS
# ============================================================================

class SimilarityMetrics:
    """Compute intra-class and inter-class similarity metrics."""
    
    def __init__(self, index: ImageRetrievalIndex):
        self.index = index
    
    def compute(self) -> Dict[str, float]:
        """Compute similarity metrics."""
        assert self.index.features is not None
        
        features = self.index.features
        labels = self.index.labels
        label_indices = self.index.label_to_indices
        
        intra_similarities = []
        inter_similarities = []
        
        # Sample for efficiency
        num_samples = min(1000, len(features))
        sample_indices = np.random.choice(len(features), num_samples, replace=False)
        
        for idx in sample_indices:
            label = labels[idx]
            feat = features[idx:idx+1]
            
            # Intra-class
            same_class_indices = [i for i in label_indices[label] if i != idx]
            if same_class_indices:
                same_feats = features[same_class_indices]
                sims = (feat @ same_feats.T).squeeze()
                if sims.dim() == 0:
                    intra_similarities.append(sims.item())
                else:
                    intra_similarities.extend(sims.tolist())
            
            # Inter-class (sample from other classes)
            other_indices = [i for i, l in enumerate(labels) if l != label]
            if other_indices:
                sample_other = np.random.choice(other_indices, min(10, len(other_indices)), replace=False)
                other_feats = features[list(sample_other)]
                sims = (feat @ other_feats.T).squeeze()
                if sims.dim() == 0:
                    inter_similarities.append(sims.item())
                else:
                    inter_similarities.extend(sims.tolist())
        
        intra_mean = np.mean(intra_similarities) if intra_similarities else 0
        inter_mean = np.mean(inter_similarities) if inter_similarities else 0
        
        return {
            'intra_class_similarity': float(intra_mean),
            'inter_class_similarity': float(inter_mean),
            'similarity_gap': float(intra_mean - inter_mean),
            'num_classes': len(label_indices)
        }


class RetrievalMetrics:
    """Compute retrieval metrics."""
    
    @staticmethod
    def recall_at_k(
        retrieved_labels: List[List[str]],
        query_labels: List[str],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute Recall@K metrics.
        
        Args:
            retrieved_labels: Labels of retrieved items per query
            query_labels: True labels for queries
            k_values: K values for recall computation
        
        Returns:
            Dictionary of R@K metrics
        """
        results = {}
        
        for k in k_values:
            correct = 0
            for query_label, retrieved in zip(query_labels, retrieved_labels):
                top_k = retrieved[:k]
                if query_label in top_k:
                    correct += 1
            
            results[f'R@{k}'] = correct / len(query_labels)
        
        return results
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_labels: List[List[str]],
        query_labels: List[str]
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        mrr = 0
        
        for query_label, retrieved in zip(query_labels, retrieved_labels):
            for rank, label in enumerate(retrieved, 1):
                if label == query_label:
                    mrr += 1.0 / rank
                    break
        
        return mrr / len(query_labels)
    
    @staticmethod
    def mean_average_precision(
        retrieved_labels: List[List[str]],
        query_labels: List[str]
    ) -> float:
        """Compute Mean Average Precision."""
        aps = []
        
        for query_label, retrieved in zip(query_labels, retrieved_labels):
            relevant = sum(1 for l in retrieved if l == query_label)
            if relevant == 0:
                aps.append(0)
                continue
            
            precision_sum = 0
            relevant_count = 0
            
            for i, label in enumerate(retrieved, 1):
                if label == query_label:
                    relevant_count += 1
                    precision_sum += relevant_count / i
            
            aps.append(precision_sum / relevant)
        
        return sum(aps) / len(aps)


# ============================================================================
# EVALUATOR
# ============================================================================

class RemoteCLIPEvaluator:
    """
    Complete evaluation system for RemoteCLIP.
    
    Evaluates:
    - Image-to-image retrieval
    - Text-to-image retrieval
    - Intra/inter-class similarity
    - Zero-shot classification
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.eval_config = config.evaluation
        self.device = config.device
        
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.preprocess: Optional[Any] = None
        
        self.image_index: Optional[ImageRetrievalIndex] = None
        
        self.results_dir = config.checkpoint_dir / "evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, checkpoint_path: Optional[Path] = None):
        """Load model for evaluation."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        
        print("\nLoading model for evaluation...")
        
        try:
            import open_clip
            
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            
            self.model = model.to(self.device)
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
        except ImportError:
            raise RuntimeError("open_clip required")
        
        # Load checkpoint
        if checkpoint_path and checkpoint_path.exists():
            from .stage2_trainer import apply_lora_to_model
            
            if hasattr(self.model, 'visual'):
                self.model.visual = apply_lora_to_model(
                    self.model.visual,
                    target_modules=self.config.training.lora_target_modules,
                    rank=self.config.training.lora_rank,
                    alpha=self.config.training.lora_alpha
                )
            
            lora_state = torch.load(checkpoint_path, map_location=self.device)
            model_state = self.model.state_dict()
            for name, param in lora_state.items():
                if name in model_state:
                    model_state[name].copy_(param)
            
            print(f"  Loaded checkpoint: {checkpoint_path}")
        
        self.model.eval()
        print("  ✓ Model loaded")
    
    def build_image_index(
        self,
        image_dir: Path,
        labels_file: Optional[Path] = None
    ):
        """Build image index from directory."""
        assert self.model is not None and self.preprocess is not None
        
        print(f"\nBuilding image index from {image_dir}...")
        
        # Collect images
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(image_dir.rglob(ext))
        
        image_paths = [str(p) for p in image_paths]
        print(f"  Found {len(image_paths)} images")
        
        # Load labels if provided
        labels = None
        if labels_file and labels_file.exists():
            with open(labels_file, 'r') as f:
                labels_data = json.load(f)
            labels = [labels_data.get(p, Path(p).parent.name) for p in image_paths]
        else:
            # Use parent directory as label
            labels = [Path(p).parent.name for p in image_paths]
        
        # Create dataset
        dataset = ImageIndexDataset(image_paths, labels, self.preprocess)
        loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
        
        # Build index
        self.image_index = ImageRetrievalIndex(self.device)
        self.image_index.build(self.model, loader)
    
    def text_to_image_retrieval(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Perform text-to-image retrieval.
        
        Args:
            queries: List of text queries
            top_k: Number of results per query
        
        Returns:
            Retrieval results with paths and scores
        """
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.image_index is not None
        
        print(f"\nText-to-image retrieval: {len(queries)} queries, top-{top_k}...")
        
        self.model.eval()
        
        with torch.no_grad():
            text_tokens = self.tokenizer(queries).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        scores, indices = self.image_index.search(text_features, top_k)
        paths = self.image_index.get_paths(indices)
        
        results = []
        for i, query in enumerate(queries):
            results.append({
                'query': query,
                'retrieved': [
                    {'path': p, 'score': float(s), 'label': self.image_index.labels[idx]}
                    for p, s, idx in zip(
                        paths[i], 
                        scores[i].tolist(), 
                        indices[i].tolist()
                    )
                ]
            })
        
        print(f"  ✓ Retrieved {top_k} images per query")
        return {'results': results}
    
    def image_to_image_retrieval(
        self,
        query_images: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Perform image-to-image retrieval.
        
        Args:
            query_images: List of query image paths
            top_k: Number of results per query
        
        Returns:
            Retrieval results
        """
        assert self.model is not None
        assert self.preprocess is not None
        assert self.image_index is not None
        
        print(f"\nImage-to-image retrieval: {len(query_images)} queries...")
        
        self.model.eval()
        
        query_features = []
        
        with torch.no_grad():
            for path in query_images:
                try:
                    if Image is None:
                        raise RuntimeError("PIL required")
                    img = Image.open(path).convert('RGB')
                    img = self.preprocess(img).unsqueeze(0).to(self.device)
                    feat = self.model.encode_image(img)
                    feat = F.normalize(feat, dim=-1)
                    query_features.append(feat)
                except Exception:
                    query_features.append(torch.zeros(1, 512, device=self.device))
        
        query_features = torch.cat(query_features, dim=0)
        scores, indices = self.image_index.search(query_features, top_k + 1)  # +1 for self
        paths = self.image_index.get_paths(indices)
        
        results = []
        for i, query_path in enumerate(query_images):
            retrieved = []
            for p, s, idx in zip(paths[i], scores[i].tolist(), indices[i].tolist()):
                if p != query_path:  # Exclude self
                    retrieved.append({
                        'path': p,
                        'score': float(s),
                        'label': self.image_index.labels[idx]
                    })
            results.append({
                'query': query_path,
                'retrieved': retrieved[:top_k]
            })
        
        return {'results': results}
    
    def compute_similarity_metrics(self) -> Dict[str, float]:
        """Compute intra/inter-class similarity metrics."""
        assert self.image_index is not None
        
        print("\nComputing similarity metrics...")
        
        metrics = SimilarityMetrics(self.image_index)
        results = metrics.compute()
        
        print(f"  Intra-class: {results['intra_class_similarity']:.3f}")
        print(f"  Inter-class: {results['inter_class_similarity']:.3f}")
        print(f"  Gap: {results['similarity_gap']:.3f}")
        
        return results
    
    def zero_shot_classification(
        self,
        test_images: List[str],
        test_labels: List[str],
        class_names: List[str],
        prompt_template: str = "a satellite image of {}"
    ) -> Dict[str, Any]:
        """
        Perform zero-shot classification.
        
        Args:
            test_images: List of test image paths
            test_labels: True labels for test images
            class_names: List of class names
            prompt_template: Template for text prompts
        
        Returns:
            Classification results and accuracy
        """
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.preprocess is not None
        
        print(f"\nZero-shot classification: {len(test_images)} images, {len(class_names)} classes...")
        
        self.model.eval()
        
        # Encode class names
        prompts = [prompt_template.format(name) for name in class_names]
        with torch.no_grad():
            text_tokens = self.tokenizer(prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        # Classify images
        predictions = []
        correct = 0
        
        with torch.no_grad():
            for img_path, true_label in zip(test_images, test_labels):
                try:
                    if Image is None:
                        raise RuntimeError("PIL required")
                    img = Image.open(img_path).convert('RGB')
                    img = self.preprocess(img).unsqueeze(0).to(self.device)
                    
                    img_feat = self.model.encode_image(img)
                    img_feat = F.normalize(img_feat, dim=-1)
                    
                    # Compute similarities
                    sims = (img_feat @ text_features.T).squeeze()
                    pred_idx = sims.argmax().item()
                    pred_label = class_names[pred_idx]
                    
                    predictions.append({
                        'path': img_path,
                        'true_label': true_label,
                        'predicted': pred_label,
                        'confidence': float(sims[pred_idx])
                    })
                    
                    if pred_label == true_label:
                        correct += 1
                except Exception:
                    predictions.append({
                        'path': img_path,
                        'true_label': true_label,
                        'predicted': 'error',
                        'confidence': 0.0
                    })
        
        accuracy = correct / len(test_images) if test_images else 0
        
        # Per-class accuracy
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for pred in predictions:
            true = pred['true_label']
            class_total[true] += 1
            if pred['predicted'] == true:
                class_correct[true] += 1
        
        per_class = {c: class_correct[c] / class_total[c] for c in class_total}
        
        print(f"  Overall accuracy: {accuracy:.3f}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'per_class_accuracy': per_class
        }
    
    def full_evaluation(
        self,
        image_dir: Path,
        test_queries: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run full evaluation suite.
        
        Args:
            image_dir: Directory with test images
            test_queries: Optional text queries for retrieval
            class_names: Optional class names for zero-shot
        
        Returns:
            Complete evaluation results
        """
        print("\n" + "="*80)
        print("FULL EVALUATION")
        print("="*80)
        
        start_time = time.time()
        results = {}
        
        # Build index
        self.build_image_index(image_dir)
        
        # Similarity metrics
        results['similarity'] = self.compute_similarity_metrics()
        
        # Text-to-image retrieval
        if test_queries:
            results['text_retrieval'] = self.text_to_image_retrieval(
                test_queries, top_k=self.eval_config.top_k_retrieval
            )
        
        # Zero-shot classification
        if class_names and self.image_index:
            test_images = self.image_index.paths
            test_labels = self.image_index.labels
            results['zero_shot'] = self.zero_shot_classification(
                test_images, test_labels, class_names
            )
        
        elapsed = time.time() - start_time
        results['evaluation_time'] = elapsed
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"evaluation_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            # Convert non-serializable items
            serializable = json.loads(json.dumps(results, default=str))
            json.dump(serializable, f, indent=2)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Results: {results_path}")
        
        return results
