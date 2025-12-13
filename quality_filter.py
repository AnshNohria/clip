#!/usr/bin/env python3
"""
QUALITY FILTER: CLIP/RemoteCLIP Similarity Scoring + Sanity Checks

This module filters the Stage 1 dataset to keep only high-quality pairs:

1. CLIP/RemoteCLIP SIMILARITY SCORING:
   - Compute image-text similarity for each pair
   - Use RemoteCLIP for domain-specific scoring (aerial/satellite)
   - Fall back to OpenAI CLIP if RemoteCLIP unavailable

2. SANITY CHECKS:
   - Object count consistency (prompt mentions vs detection)
   - Layout consistency (spatial terms match visual layout)
   - Image quality metrics (blur, contrast, artifacts)

3. FILTERING:
   - Keep top-X% by composite quality score
   - Separate thresholds for real vs synthetic
   - Generate quality report

Output:
   - filtered_dataset.json: Pairs that passed quality threshold
   - quality_report.json: Detailed per-pair scoring
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Check for dependencies
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ============================================================================
# CONFIGURATION
# ============================================================================

class QualityConfig:
    """Configuration for quality filtering"""
    
    # Input/Output
    INPUT_DATASET = Path("datasets/stage1_corpus/dataset.json")
    OUTPUT_DIR = Path("datasets/stage1_corpus/filtered")
    
    # Scoring weights
    CLIP_WEIGHT = 0.50  # Weight for CLIP similarity
    SANITY_WEIGHT = 0.25  # Weight for sanity checks
    IMAGE_QUALITY_WEIGHT = 0.25  # Weight for image quality
    
    # Thresholds
    MIN_CLIP_SCORE_REAL = 0.20  # Minimum CLIP score for real pairs
    MIN_CLIP_SCORE_SYNTHETIC = 0.25  # Higher threshold for synthetic
    KEEP_TOP_PERCENT = 0.70  # Keep top 70% by composite score
    
    # RemoteCLIP settings
    REMOTECLIP_MODEL = "ViT-B-32"  # RemoteCLIP variant
    REMOTECLIP_CHECKPOINT = "RemoteCLIP_checkpoints"  # Local checkpoint dir
    FALLBACK_TO_OPENAI_CLIP = True  # Use OpenAI CLIP if RemoteCLIP unavailable
    
    # Image quality thresholds
    MIN_BRIGHTNESS = 20  # Minimum average brightness (0-255)
    MAX_BRIGHTNESS = 235  # Maximum average brightness
    MIN_CONTRAST = 30  # Minimum std deviation of pixels
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# REMOTECLIP SCORER
# ============================================================================

class RemoteCLIPScorer:
    """
    Compute image-text similarity using RemoteCLIP or fallback CLIP.
    
    RemoteCLIP is specifically trained on remote sensing imagery and
    provides better alignment for aerial/satellite images.
    """
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.model_type = None
        
        self._init_model()
    
    def _init_model(self):
        """Initialize CLIP model (RemoteCLIP preferred)"""
        if not HAS_TORCH:
            print("⚠ PyTorch not available, CLIP scoring disabled")
            return
        
        # Try RemoteCLIP first
        if self._try_load_remoteclip():
            return
        
        # Fallback to OpenAI CLIP
        if self.config.FALLBACK_TO_OPENAI_CLIP:
            self._load_openai_clip()
    
    def _try_load_remoteclip(self) -> bool:
        """Try to load RemoteCLIP model"""
        try:
            import open_clip
            
            checkpoint_dir = Path(self.config.REMOTECLIP_CHECKPOINT)
            
            # Check for local checkpoint
            if checkpoint_dir.exists():
                # Look for RemoteCLIP checkpoint
                checkpoints = list(checkpoint_dir.rglob("*.pt"))
                checkpoints.extend(checkpoint_dir.rglob("*.bin"))
                
                if checkpoints:
                    print(f"[QualityFilter] Loading RemoteCLIP from {checkpoints[0]}...")
                    
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        'ViT-B-32',
                        pretrained=str(checkpoints[0])
                    )
                    self.model = model.to(DEVICE).eval()
                    self.preprocess = preprocess
                    self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
                    self.model_type = "RemoteCLIP"
                    print("✓ RemoteCLIP loaded successfully")
                    return True
            
            # Try loading from HuggingFace
            print("[QualityFilter] Attempting to load RemoteCLIP from HuggingFace...")
            
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
                self.model = model.to(DEVICE).eval()
                self.preprocess = preprocess
                self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
                self.model_type = "RemoteCLIP"
                print("✓ RemoteCLIP loaded from HuggingFace")
                return True
                
            except Exception as e:
                print(f"  Could not load from HuggingFace: {e}")
                return False
                
        except ImportError:
            print("  open_clip not installed, cannot load RemoteCLIP")
            return False
        except Exception as e:
            print(f"  Error loading RemoteCLIP: {e}")
            return False
    
    def _load_openai_clip(self):
        """Load OpenAI CLIP as fallback"""
        try:
            import clip
            
            print("[QualityFilter] Loading OpenAI CLIP (fallback)...")
            self.model, self.preprocess = clip.load("ViT-B/32", device=DEVICE)
            self.tokenizer = clip.tokenize
            self.model_type = "OpenAI-CLIP"
            print("✓ OpenAI CLIP loaded")
            
        except ImportError:
            # Try open_clip with pretrained OpenAI weights
            try:
                import open_clip
                
                print("[QualityFilter] Loading CLIP via open_clip...")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    pretrained='openai'
                )
                self.model = model.to(DEVICE).eval()
                self.preprocess = preprocess
                self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
                self.model_type = "OpenCLIP"
                print("✓ OpenCLIP loaded")
                
            except Exception as e:
                print(f"⚠ Could not load any CLIP model: {e}")
    
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """
        Compute cosine similarity between image and text.
        
        Returns similarity score in [0, 1].
        """
        if self.model is None:
            return 0.5  # Neutral score if no model
        
        try:
            # Preprocess image
            image_input = self.preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Tokenize text
            if self.model_type == "OpenAI-CLIP":
                text_input = self.tokenizer([text], truncate=True).to(DEVICE)
            else:
                text_input = self.tokenizer([text]).to(DEVICE)
            
            # Compute embeddings
            with torch.no_grad():
                if self.model_type == "OpenAI-CLIP":
                    image_features = self.model.encode_image(image_input)
                    text_features = self.model.encode_text(text_input)
                else:
                    image_features = self.model.encode_image(image_input)
                    text_features = self.model.encode_text(text_input)
                
                # Normalize
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Cosine similarity
                similarity = (image_features @ text_features.T).item()
            
            # Convert to [0, 1] range (CLIP similarity is in [-1, 1])
            return (similarity + 1) / 2
            
        except Exception as e:
            print(f"  ⚠ Similarity computation error: {e}")
            return 0.5
    
    def batch_compute_similarity(self, image_paths: List[Path], 
                                  texts: List[str],
                                  batch_size: int = 32) -> List[float]:
        """Compute similarity for multiple pairs efficiently"""
        if self.model is None:
            return [0.5] * len(image_paths)
        
        similarities = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            # Load and preprocess images
            images = []
            valid_indices = []
            for idx, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(self.preprocess(img))
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"  ⚠ Could not load {path}: {e}")
            
            if not images:
                similarities.extend([0.0] * len(batch_paths))
                continue
            
            # Stack images
            image_input = torch.stack(images).to(DEVICE)
            
            # Tokenize texts
            valid_texts = [batch_texts[i] for i in valid_indices]
            if self.model_type == "OpenAI-CLIP":
                text_input = self.tokenizer(valid_texts, truncate=True).to(DEVICE)
            else:
                text_input = self.tokenizer(valid_texts).to(DEVICE)
            
            # Compute embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Diagonal similarities (each image with its text)
                batch_sim = (image_features * text_features).sum(dim=-1)
            
            # Map back to original order
            batch_results = [0.0] * len(batch_paths)
            for sim_idx, orig_idx in enumerate(valid_indices):
                batch_results[orig_idx] = (batch_sim[sim_idx].item() + 1) / 2
            
            similarities.extend(batch_results)
        
        return similarities


# ============================================================================
# SANITY CHECKER
# ============================================================================

class SanityChecker:
    """
    Perform sanity checks on image-caption pairs.
    
    Checks:
    1. Object count consistency
    2. Spatial layout consistency
    3. Scene type consistency
    """
    
    # Number words to integers
    NUMBER_WORDS = {
        'one': 1, 'a': 1, 'an': 1, 'single': 1,
        'two': 2, 'couple': 2, 'pair': 2,
        'three': 3, 'few': 3,
        'four': 4, 'several': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'many': 10, 'numerous': 10, 'multiple': 5
    }
    
    # Spatial keywords
    SPATIAL_KEYWORDS = {
        'left': ['left'],
        'right': ['right'],
        'top': ['top', 'upper', 'north'],
        'bottom': ['bottom', 'lower', 'south'],
        'center': ['center', 'middle', 'central'],
        'corner': ['corner']
    }
    
    # Scene types
    SCENE_TYPES = {
        'urban': ['building', 'road', 'street', 'urban', 'city', 'residential', 'commercial'],
        'vegetation': ['tree', 'forest', 'vegetation', 'grass', 'park', 'green'],
        'water': ['water', 'river', 'lake', 'pond', 'ocean', 'sea', 'pool'],
        'agricultural': ['farm', 'field', 'crop', 'agricultural', 'farmland'],
        'industrial': ['factory', 'industrial', 'warehouse', 'storage']
    }
    
    def check_consistency(self, caption: str, metadata: Dict) -> Dict:
        """
        Check various consistency metrics.
        
        Returns dict with scores for each check.
        """
        results = {
            'object_count_score': 1.0,
            'spatial_score': 1.0,
            'scene_type_score': 1.0,
            'caption_quality_score': 1.0
        }
        
        caption_lower = caption.lower()
        
        # 1. Object count consistency
        if metadata.get('detection_counts'):
            results['object_count_score'] = self._check_object_counts(
                caption_lower, metadata['detection_counts']
            )
        
        # 2. Spatial consistency
        if metadata.get('spatial_layout'):
            results['spatial_score'] = self._check_spatial_consistency(
                caption_lower, metadata['spatial_layout']
            )
        
        # 3. Caption quality (basic checks)
        results['caption_quality_score'] = self._check_caption_quality(caption)
        
        # Composite sanity score
        results['composite_sanity'] = (
            results['object_count_score'] * 0.3 +
            results['spatial_score'] * 0.3 +
            results['caption_quality_score'] * 0.4
        )
        
        return results
    
    def _check_object_counts(self, caption: str, detection_counts: Dict) -> float:
        """Check if mentioned counts match detection counts"""
        # Extract numbers from caption
        mentioned_counts = {}
        
        # Find numeric mentions
        for word, num in self.NUMBER_WORDS.items():
            if word in caption:
                # Try to find what object the number refers to
                pattern = rf'{word}\s+(\w+)'
                matches = re.findall(pattern, caption)
                for match in matches:
                    mentioned_counts[match] = num
        
        # Also find digit mentions
        digit_pattern = r'(\d+)\s+(\w+)'
        for match in re.finditer(digit_pattern, caption):
            num = int(match.group(1))
            obj = match.group(2)
            mentioned_counts[obj] = num
        
        if not mentioned_counts or not detection_counts:
            return 0.8  # Neutral-ish score
        
        # Compare mentioned vs detected
        score = 1.0
        for obj, mentioned_num in mentioned_counts.items():
            # Find matching detection
            detected_num = 0
            for det_obj, det_count in detection_counts.items():
                if obj in det_obj.lower() or det_obj.lower() in obj:
                    detected_num = det_count
                    break
            
            if detected_num > 0:
                # Penalize mismatch
                ratio = min(mentioned_num, detected_num) / max(mentioned_num, detected_num)
                score = min(score, ratio)
        
        return score
    
    def _check_spatial_consistency(self, caption: str, spatial_layout: str) -> float:
        """Check if spatial terms in caption match actual layout"""
        spatial_mentions = []
        
        for direction, keywords in self.SPATIAL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in caption:
                    spatial_mentions.append(direction)
        
        if not spatial_mentions:
            return 0.9  # No spatial claims = OK
        
        # Check against layout (if available)
        layout_lower = spatial_layout.lower() if spatial_layout else ""
        
        matches = 0
        for mention in spatial_mentions:
            if mention in layout_lower:
                matches += 1
        
        if len(spatial_mentions) > 0:
            return 0.5 + 0.5 * (matches / len(spatial_mentions))
        
        return 0.8
    
    def _check_caption_quality(self, caption: str) -> float:
        """Basic caption quality checks"""
        score = 1.0
        
        # Too short
        if len(caption) < 20:
            score -= 0.3
        
        # Too long (possibly garbled)
        if len(caption) > 500:
            score -= 0.2
        
        # Repeated words (sign of generation issues)
        words = caption.lower().split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 0.3
        
        # Has basic sentence structure
        if not any(c in caption for c in '.!?'):
            score -= 0.1
        
        return max(0.0, score)


# ============================================================================
# IMAGE QUALITY ANALYZER
# ============================================================================

class ImageQualityAnalyzer:
    """
    Analyze image quality metrics.
    
    Checks:
    1. Brightness (not too dark/bright)
    2. Contrast (not too flat)
    3. Blur detection
    4. Artifact detection
    """
    
    def __init__(self, config: QualityConfig):
        self.config = config
    
    def analyze(self, image: Image.Image) -> Dict:
        """
        Analyze image quality.
        
        Returns dict with quality metrics.
        """
        results = {
            'brightness_score': 1.0,
            'contrast_score': 1.0,
            'blur_score': 1.0,
            'artifact_score': 1.0
        }
        
        # Convert to numpy
        img_array = np.array(image.convert('L'))  # Grayscale for analysis
        
        # 1. Brightness check
        mean_brightness = np.mean(img_array)
        if mean_brightness < self.config.MIN_BRIGHTNESS:
            results['brightness_score'] = mean_brightness / self.config.MIN_BRIGHTNESS
        elif mean_brightness > self.config.MAX_BRIGHTNESS:
            results['brightness_score'] = (255 - mean_brightness) / (255 - self.config.MAX_BRIGHTNESS)
        
        # 2. Contrast check (standard deviation)
        std_dev = np.std(img_array)
        if std_dev < self.config.MIN_CONTRAST:
            results['contrast_score'] = std_dev / self.config.MIN_CONTRAST
        else:
            results['contrast_score'] = min(1.0, std_dev / 60)  # Normalize
        
        # 3. Blur detection (Laplacian variance)
        try:
            from PIL import ImageFilter
            laplacian = image.convert('L').filter(ImageFilter.FIND_EDGES)
            lap_array = np.array(laplacian)
            lap_var = np.var(lap_array)
            
            # Higher variance = sharper image
            results['blur_score'] = min(1.0, lap_var / 500)
        except:
            results['blur_score'] = 0.8  # Default
        
        # 4. Artifact detection (look for unusual patterns)
        # Check for repeated patterns that might indicate generation artifacts
        h, w = img_array.shape
        if h > 16 and w > 16:
            # Check patches
            patches = []
            for y in range(0, h - 16, 16):
                for x in range(0, w - 16, 16):
                    patch = img_array[y:y+16, x:x+16]
                    patches.append(patch.mean())
            
            # Too uniform might indicate artifacts
            patch_var = np.var(patches)
            if patch_var < 10:
                results['artifact_score'] = 0.5
            else:
                results['artifact_score'] = min(1.0, patch_var / 100)
        
        # Composite image quality
        results['composite_image_quality'] = (
            results['brightness_score'] * 0.25 +
            results['contrast_score'] * 0.30 +
            results['blur_score'] * 0.30 +
            results['artifact_score'] * 0.15
        )
        
        results['brightness'] = float(mean_brightness)
        results['contrast'] = float(std_dev)
        
        return results


# ============================================================================
# QUALITY FILTER PIPELINE
# ============================================================================

@dataclass
class QualityReport:
    """Quality report for a single pair"""
    image_id: str
    source: str
    clip_score: float
    sanity_score: float
    image_quality_score: float
    composite_score: float
    passed: bool
    details: Dict


class QualityFilterPipeline:
    """
    Main pipeline for quality filtering.
    """
    
    def __init__(self, config: QualityConfig = None):
        self.config = config or QualityConfig()
        
        # Initialize components
        self.clip_scorer = RemoteCLIPScorer(self.config)
        self.sanity_checker = SanityChecker()
        self.image_analyzer = ImageQualityAnalyzer(self.config)
        
        # Results storage
        self.reports: List[QualityReport] = []
        self.passed_pairs = []
        self.failed_pairs = []
    
    def load_dataset(self) -> Dict:
        """Load the Stage 1 dataset"""
        if not self.config.INPUT_DATASET.exists():
            raise FileNotFoundError(f"Dataset not found: {self.config.INPUT_DATASET}")
        
        with open(self.config.INPUT_DATASET, 'r') as f:
            return json.load(f)
    
    def score_pair(self, pair: Dict, dataset_dir: Path) -> QualityReport:
        """
        Score a single image-caption pair.
        
        Returns QualityReport with all scores.
        """
        image_id = pair['image_id']
        image_path = dataset_dir / pair['image_path']
        caption = pair['caption']
        source = pair['source']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return QualityReport(
                image_id=image_id,
                source=source,
                clip_score=0.0,
                sanity_score=0.0,
                image_quality_score=0.0,
                composite_score=0.0,
                passed=False,
                details={'error': str(e)}
            )
        
        # 1. CLIP similarity
        clip_score = self.clip_scorer.compute_similarity(image, caption)
        
        # 2. Sanity checks
        sanity_results = self.sanity_checker.check_consistency(caption, pair)
        sanity_score = sanity_results['composite_sanity']
        
        # 3. Image quality
        quality_results = self.image_analyzer.analyze(image)
        image_quality_score = quality_results['composite_image_quality']
        
        # 4. Composite score
        composite_score = (
            clip_score * self.config.CLIP_WEIGHT +
            sanity_score * self.config.SANITY_WEIGHT +
            image_quality_score * self.config.IMAGE_QUALITY_WEIGHT
        )
        
        # 5. Determine pass/fail
        if source == 'synthetic':
            min_clip = self.config.MIN_CLIP_SCORE_SYNTHETIC
        else:
            min_clip = self.config.MIN_CLIP_SCORE_REAL
        
        passed = clip_score >= min_clip and composite_score >= 0.3
        
        return QualityReport(
            image_id=image_id,
            source=source,
            clip_score=clip_score,
            sanity_score=sanity_score,
            image_quality_score=image_quality_score,
            composite_score=composite_score,
            passed=passed,
            details={
                'sanity_details': sanity_results,
                'image_quality_details': quality_results
            }
        )
    
    def filter_dataset(self) -> Dict:
        """
        Filter the entire dataset and return filtered version.
        """
        print("\n" + "="*80)
        print("QUALITY FILTERING PIPELINE")
        print("="*80)
        
        # Load dataset
        print("\n[1/4] Loading dataset...")
        dataset = self.load_dataset()
        pairs = dataset['pairs']
        dataset_dir = self.config.INPUT_DATASET.parent
        
        print(f"  Loaded {len(pairs)} pairs")
        print(f"  CLIP model: {self.clip_scorer.model_type or 'None'}")
        
        # Score all pairs
        print("\n[2/4] Scoring pairs...")
        
        for i, pair in enumerate(pairs):
            if i % 10 == 0:
                print(f"  Processing {i+1}/{len(pairs)}...", end='\r')
            
            report = self.score_pair(pair, dataset_dir)
            self.reports.append(report)
            
            if report.passed:
                self.passed_pairs.append(pair)
            else:
                self.failed_pairs.append(pair)
        
        print(f"\n  Scored {len(pairs)} pairs")
        
        # Apply top-X% filtering
        print("\n[3/4] Applying top-X% filter...")
        
        # Sort by composite score
        scored_pairs = list(zip(self.reports, pairs))
        scored_pairs.sort(key=lambda x: x[0].composite_score, reverse=True)
        
        # Keep top X%
        keep_count = int(len(scored_pairs) * self.config.KEEP_TOP_PERCENT)
        top_pairs = scored_pairs[:keep_count]
        
        final_pairs = [p[1] for p in top_pairs]
        final_reports = [p[0] for p in top_pairs]
        
        print(f"  Keeping top {self.config.KEEP_TOP_PERCENT*100:.0f}%: {keep_count} pairs")
        
        # Create filtered dataset
        filtered_dataset = {
            'name': dataset['name'] + '_filtered',
            'version': dataset.get('version', '1.0.0'),
            'created_at': datetime.now().isoformat(),
            'filtered_from': str(self.config.INPUT_DATASET),
            'filter_config': {
                'clip_weight': self.config.CLIP_WEIGHT,
                'sanity_weight': self.config.SANITY_WEIGHT,
                'image_quality_weight': self.config.IMAGE_QUALITY_WEIGHT,
                'min_clip_score_real': self.config.MIN_CLIP_SCORE_REAL,
                'min_clip_score_synthetic': self.config.MIN_CLIP_SCORE_SYNTHETIC,
                'keep_top_percent': self.config.KEEP_TOP_PERCENT,
                'clip_model': self.clip_scorer.model_type
            },
            'stats': {
                'original_count': len(pairs),
                'filtered_count': len(final_pairs),
                'removed_count': len(pairs) - len(final_pairs),
                'avg_clip_score': np.mean([r.clip_score for r in final_reports]),
                'avg_composite_score': np.mean([r.composite_score for r in final_reports])
            },
            'pairs': final_pairs
        }
        
        return filtered_dataset
    
    def save_results(self, filtered_dataset: Dict):
        """Save filtered dataset and quality report"""
        print("\n[4/4] Saving results...")
        
        # Save filtered dataset
        filtered_path = self.config.OUTPUT_DIR / "filtered_dataset.json"
        with open(filtered_path, 'w') as f:
            json.dump(filtered_dataset, f, indent=2)
        print(f"  ✓ Filtered dataset: {filtered_path}")
        
        # Save quality report
        report_data = {
            'created_at': datetime.now().isoformat(),
            'summary': {
                'total_pairs': len(self.reports),
                'passed_count': len(self.passed_pairs),
                'failed_count': len(self.failed_pairs),
                'pass_rate': len(self.passed_pairs) / len(self.reports) if self.reports else 0
            },
            'score_distribution': {
                'clip_scores': {
                    'min': min(r.clip_score for r in self.reports),
                    'max': max(r.clip_score for r in self.reports),
                    'mean': np.mean([r.clip_score for r in self.reports]),
                    'std': np.std([r.clip_score for r in self.reports])
                },
                'composite_scores': {
                    'min': min(r.composite_score for r in self.reports),
                    'max': max(r.composite_score for r in self.reports),
                    'mean': np.mean([r.composite_score for r in self.reports]),
                    'std': np.std([r.composite_score for r in self.reports])
                }
            },
            'per_pair_reports': [asdict(r) for r in self.reports]
        }
        
        report_path = self.config.OUTPUT_DIR / "quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"  ✓ Quality report: {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("FILTERING SUMMARY")
        print("="*80)
        print(f"  Original pairs: {len(self.reports)}")
        print(f"  Filtered pairs: {filtered_dataset['stats']['filtered_count']}")
        print(f"  Removed: {filtered_dataset['stats']['removed_count']}")
        print(f"  Avg CLIP score: {filtered_dataset['stats']['avg_clip_score']:.3f}")
        print(f"  Avg composite score: {filtered_dataset['stats']['avg_composite_score']:.3f}")
    
    def run(self):
        """Run the complete quality filtering pipeline"""
        filtered_dataset = self.filter_dataset()
        self.save_results(filtered_dataset)
        
        return filtered_dataset


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Main entry point for quality filtering"""
    print("\n" + "="*80)
    print("QUALITY FILTER: CLIP/RemoteCLIP Scoring + Sanity Checks")
    print("="*80)
    
    config = QualityConfig()
    pipeline = QualityFilterPipeline(config)
    
    filtered_dataset = pipeline.run()
    
    print("\n" + "="*80)
    print("QUALITY FILTERING COMPLETE")
    print("="*80)
    print(f"Filtered dataset: {config.OUTPUT_DIR / 'filtered_dataset.json'}")
    print("\nNext steps:")
    print("  1. Review quality_report.json for scoring details")
    print("  2. Run stage2_lora_finetune.py for synthetic-heavy pretraining")


if __name__ == "__main__":
    main()
