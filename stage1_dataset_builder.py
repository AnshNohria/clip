#!/usr/bin/env python3
"""
STAGE 1: Build and Freeze the Augmented + Synthetic Dataset

This module constructs the full training corpus for Stage 2 fine-tuning:

A) REAL AUGMENTED SET (high-trust supervision):
   - Standard geometric/photometric augmentations (crop, flip, color jitter)
   - Enriched captions via M2B + B2C algorithmic analysis + Qwen2-VL VLM
   - Object-level "zoom-in" crops for dense scenes
   - Explicitly tagged as "real"

B) SYNTHETIC SET (from 5-stage pipeline):
   - Real-ESRGAN → Qwen2-VL → Grounding DINO → SAM → SD3.5 pipeline
   - Full metadata stored: layout, counts, prompts, source image
   - Quality filtered via CLIP/RemoteCLIP similarity
   - Explicitly tagged as "synthetic"

Output Format:
- dataset.json: Master index with all image-caption pairs
- images/real_aug/: Real augmented images
- images/synthetic/: Synthetic generated images
- metadata/: Per-image JSON with full provenance
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Optional GPU check (not required for dataset building)
try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False
    HAS_CUDA = False

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

class DatasetConfig:
    """Configuration for Stage 1 dataset building"""
    
    # Input paths
    RSICD_IMAGES_DIR = Path("datasets/rsicd_images")
    RSICD_CAPTIONS_JSON = Path("datasets/UCM_captions/dataset.json")  # Fallback captions
    
    # Output paths
    OUTPUT_DIR = Path("datasets/stage1_corpus")
    REAL_AUG_DIR = Path("datasets/stage1_corpus/images/real_aug")
    SYNTHETIC_DIR = Path("datasets/stage1_corpus/images/synthetic")
    METADATA_DIR = Path("datasets/stage1_corpus/metadata")
    
    # Augmentation settings
    NUM_AUGMENTATIONS_PER_IMAGE = 5  # Number of augmented versions per real image
    ENABLE_ZOOM_CROPS = True  # Generate object-level zoom crops
    ZOOM_CROP_MIN_SIZE = 128  # Minimum crop size in pixels
    
    # Synthetic generation settings
    NUM_SYNTHETIC_PER_IMAGE = 2  # Number of synthetic variants per source image
    SYNTHETIC_PIPELINE_AVAILABLE = False  # Set True if run_hpc_pipeline is available
    
    # Quality thresholds
    MIN_CLIP_SIMILARITY = 0.25  # Minimum CLIP score to keep synthetic pair
    KEEP_TOP_PERCENT = 0.70  # Keep top 70% of synthetic pairs by quality
    
    # Caption enrichment
    USE_VLM_CAPTIONS = True  # Use Qwen2-VL for captions (requires GPU)
    USE_M2B_B2C = True  # Use algorithmic M2B+B2C enrichment
    
    def __init__(self):
        # Create output directories
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.REAL_AUG_DIR.mkdir(parents=True, exist_ok=True)
        self.SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
        self.METADATA_DIR.mkdir(parents=True, exist_ok=True)


class DataSource(Enum):
    """Tag for data source type"""
    REAL = "real"
    REAL_AUGMENTED = "real_augmented"
    SYNTHETIC = "synthetic"


@dataclass
class ImageCaptionPair:
    """A single image-caption pair with full metadata"""
    image_id: str
    image_path: str
    caption: str
    source: str  # DataSource value
    source_image_id: Optional[str] = None  # For augmented/synthetic: original image
    
    # Caption provenance
    caption_sources: List[str] = field(default_factory=list)  # e.g., ["qwen2vl", "m2b_b2c"]
    
    # Augmentation info (for real_augmented)
    augmentation_type: Optional[str] = None
    augmentation_params: Optional[Dict] = None
    
    # Synthetic info
    synthetic_prompt: Optional[str] = None
    detection_counts: Optional[Dict] = None
    spatial_layout: Optional[str] = None
    
    # Quality scores
    clip_similarity: Optional[float] = None
    quality_score: Optional[float] = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# M2B + B2C CAPTION ENRICHMENT (Algorithmic, No ML)
# ============================================================================

class MaskToBoxExtractor:
    """
    M2B: Extract regions from images using edge detection and contour analysis.
    Pure algorithmic approach - no ML models required.
    """
    
    def __init__(self, min_region_size: int = 500, max_regions: int = 20):
        self.min_region_size = min_region_size
        self.max_regions = max_regions
    
    def extract_regions(self, image: Image.Image) -> List[Dict]:
        """Extract regions using edge detection + connected components"""
        # Convert to grayscale
        gray = image.convert('L')
        
        # Apply edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.MaxFilter(3))
        
        # Threshold to binary
        edge_array = np.array(edges)
        threshold = np.percentile(edge_array, 85)
        binary = (edge_array > threshold).astype(np.uint8)
        
        # Find connected components (simple flood-fill approach)
        regions = self._find_connected_components(binary, image)
        
        # Sort by area and limit
        regions.sort(key=lambda r: r['area'], reverse=True)
        return regions[:self.max_regions]
    
    def _find_connected_components(self, binary: np.ndarray, image: Image.Image) -> List[Dict]:
        """Simple connected component analysis"""
        h, w = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        regions = []
        
        # Grid-based region detection (3x3 grid)
        grid_h, grid_w = h // 3, w // 3
        
        for gy in range(3):
            for gx in range(3):
                # Get grid cell bounds
                y1, y2 = gy * grid_h, (gy + 1) * grid_h
                x1, x2 = gx * grid_w, (gx + 1) * grid_w
                
                # Analyze this grid cell
                cell = binary[y1:y2, x1:x2]
                edge_density = np.mean(cell)
                
                if edge_density > 0.05:  # Has significant edges
                    # Get color info from original image
                    cell_img = image.crop((x1, y1, x2, y2))
                    colors = self._analyze_colors(cell_img)
                    
                    # Grid position name
                    positions = [
                        ["top-left", "top-center", "top-right"],
                        ["middle-left", "center", "middle-right"],
                        ["bottom-left", "bottom-center", "bottom-right"]
                    ]
                    
                    regions.append({
                        'bbox': (x1, y1, x2, y2),
                        'area': (x2 - x1) * (y2 - y1),
                        'position': positions[gy][gx],
                        'edge_density': float(edge_density),
                        'dominant_colors': colors,
                        'texture_score': float(edge_density * 10)
                    })
        
        return regions
    
    def _analyze_colors(self, image: Image.Image) -> List[str]:
        """Analyze dominant colors in a region"""
        # Resize for faster processing
        small = image.resize((32, 32))
        pixels = np.array(small)
        
        if len(pixels.shape) < 3:
            return ["gray"]
        
        # Convert to HSV-like analysis
        r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
        
        colors = []
        
        # Check for vegetation (green dominant)
        green_ratio = np.mean(g) / (np.mean(r) + np.mean(b) + 1)
        if green_ratio > 0.6 and np.mean(g) > 80:
            colors.append("green")
        
        # Check for water (blue dominant)
        blue_ratio = np.mean(b) / (np.mean(r) + np.mean(g) + 1)
        if blue_ratio > 0.5 and np.mean(b) > 100:
            colors.append("blue")
        
        # Check for urban/roads (gray/dark)
        brightness = (np.mean(r) + np.mean(g) + np.mean(b)) / 3
        color_variance = np.std([np.mean(r), np.mean(g), np.mean(b)])
        if color_variance < 20 and brightness < 150:
            colors.append("gray")
        
        # Check for buildings (brown/beige/white)
        if brightness > 150 and color_variance < 30:
            colors.append("light")
        
        if not colors:
            colors.append("mixed")
        
        return colors


class BoxToCaptionGenerator:
    """
    B2C: Generate captions from detected regions.
    Maps visual features to natural language descriptions.
    """
    
    # Color to scene element mapping
    COLOR_MAPPINGS = {
        "green": ["vegetation", "trees", "grass", "forest", "park"],
        "blue": ["water", "pond", "river", "pool"],
        "gray": ["roads", "pavement", "concrete", "urban area"],
        "light": ["buildings", "rooftops", "structures"],
        "brown": ["bare ground", "dirt", "farmland"],
        "mixed": ["mixed development", "suburban area"]
    }
    
    # Position descriptions
    POSITION_PHRASES = {
        "top-left": "in the upper left area",
        "top-center": "at the top",
        "top-right": "in the upper right area",
        "middle-left": "on the left side",
        "center": "in the central area",
        "middle-right": "on the right side",
        "bottom-left": "in the lower left area",
        "bottom-center": "at the bottom",
        "bottom-right": "in the lower right area"
    }
    
    def generate_caption(self, regions: List[Dict], image_size: Tuple[int, int]) -> str:
        """Generate a descriptive caption from extracted regions"""
        if not regions:
            return "Aerial view of terrain."
        
        # Analyze overall scene
        scene_type = self._determine_scene_type(regions)
        
        # Build description parts
        descriptions = []
        
        # Scene opener
        descriptions.append(f"Aerial view showing {scene_type}")
        
        # Describe significant regions
        significant_regions = [r for r in regions if r['edge_density'] > 0.08]
        
        for region in significant_regions[:5]:  # Limit to 5 most significant
            colors = region['dominant_colors']
            position = region['position']
            
            # Get element description from colors
            elements = []
            for color in colors:
                if color in self.COLOR_MAPPINGS:
                    elements.extend(self.COLOR_MAPPINGS[color][:1])
            
            if elements:
                element_str = elements[0]
                pos_phrase = self.POSITION_PHRASES.get(position, "")
                if pos_phrase:
                    descriptions.append(f"{element_str} {pos_phrase}")
        
        # Combine into final caption
        if len(descriptions) > 1:
            caption = f"{descriptions[0]} with {', '.join(descriptions[1:3])}"
            if len(descriptions) > 3:
                caption += f", and {descriptions[3]}"
        else:
            caption = descriptions[0]
        
        return caption + "."
    
    def _determine_scene_type(self, regions: List[Dict]) -> str:
        """Determine overall scene type from regions"""
        all_colors = []
        for r in regions:
            all_colors.extend(r['dominant_colors'])
        
        color_counts = {}
        for c in all_colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        # Determine dominant characteristic
        if color_counts.get('green', 0) > len(regions) * 0.5:
            return "vegetation-dominated landscape"
        elif color_counts.get('gray', 0) > len(regions) * 0.4:
            return "urban/developed area"
        elif color_counts.get('blue', 0) > len(regions) * 0.3:
            return "area with water features"
        elif color_counts.get('light', 0) > len(regions) * 0.4:
            return "residential or commercial area"
        else:
            return "mixed-use terrain"


class CaptionEnricher:
    """
    Combines M2B + B2C for caption enrichment.
    Optionally integrates VLM captions.
    """
    
    def __init__(self):
        self.m2b = MaskToBoxExtractor()
        self.b2c = BoxToCaptionGenerator()
    
    def enrich_caption(self, image: Image.Image, base_caption: str = "") -> Dict:
        """
        Enrich a caption using M2B + B2C analysis.
        
        Returns:
            Dict with 'enriched_caption', 'regions', 'scene_type'
        """
        # Extract regions (M2B)
        regions = self.m2b.extract_regions(image)
        
        # Generate description (B2C)
        m2b_caption = self.b2c.generate_caption(regions, image.size)
        
        # Determine scene type
        scene_type = self.b2c._determine_scene_type(regions)
        
        # Combine with base caption if provided
        if base_caption:
            enriched = f"{base_caption} {m2b_caption}"
        else:
            enriched = m2b_caption
        
        return {
            'enriched_caption': enriched,
            'regions': regions,
            'scene_type': scene_type,
            'm2b_caption': m2b_caption
        }


# ============================================================================
# IMAGE AUGMENTATION
# ============================================================================

class ImageAugmentor:
    """Standard geometric and photometric augmentations for real images"""
    
    def __init__(self, output_size: Tuple[int, int] = (512, 512)):
        self.output_size = output_size
    
    def augment(self, image: Image.Image, aug_type: str = "random") -> Tuple[Image.Image, Dict]:
        """
        Apply augmentation and return (augmented_image, params_dict)
        """
        if aug_type == "random":
            aug_type = random.choice([
                "horizontal_flip", "vertical_flip", "rotate",
                "brightness", "contrast", "crop", "color_jitter"
            ])
        
        params: Dict[str, Any] = {"type": aug_type}
        
        if aug_type == "horizontal_flip":
            result = ImageOps.mirror(image)
            
        elif aug_type == "vertical_flip":
            result = ImageOps.flip(image)
            
        elif aug_type == "rotate":
            angle = random.choice([90, 180, 270])
            params["angle"] = angle
            result = image.rotate(angle, expand=True)
            
        elif aug_type == "brightness":
            factor = random.uniform(0.7, 1.3)
            params["factor"] = factor
            enhancer = ImageEnhance.Brightness(image)
            result = enhancer.enhance(factor)
            
        elif aug_type == "contrast":
            factor = random.uniform(0.7, 1.3)
            params["factor"] = factor
            enhancer = ImageEnhance.Contrast(image)
            result = enhancer.enhance(factor)
            
        elif aug_type == "color_jitter":
            # Apply multiple color adjustments
            img = image
            for enhance_type in [ImageEnhance.Color, ImageEnhance.Brightness, ImageEnhance.Contrast]:
                factor = random.uniform(0.8, 1.2)
                img = enhance_type(img).enhance(factor)
            params["jitter"] = True
            result = img
            
        elif aug_type == "crop":
            # Random crop (80-95% of image)
            w, h = image.size
            crop_ratio = random.uniform(0.8, 0.95)
            new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            params["crop_box"] = (left, top, left + new_w, top + new_h)
            result = image.crop((left, top, left + new_w, top + new_h))
        else:
            result = image
        
        # Resize to output size
        result = result.resize(self.output_size, Image.Resampling.LANCZOS)
        
        return result, params
    
    def generate_zoom_crops(self, image: Image.Image, regions: List[Dict], 
                           min_size: int = 128) -> List[Tuple[Image.Image, Dict]]:
        """
        Generate object-level zoom crops from detected regions.
        Returns list of (cropped_image, crop_info)
        """
        crops = []
        
        for region in regions:
            bbox = region.get('bbox')
            if not bbox:
                continue
            
            x1, y1, x2, y2 = bbox
            crop_w, crop_h = x2 - x1, y2 - y1
            
            # Skip if too small
            if crop_w < min_size or crop_h < min_size:
                continue
            
            # Add padding (20% on each side)
            pad_x = int(crop_w * 0.2)
            pad_y = int(crop_h * 0.2)
            
            # Clamp to image bounds
            w, h = image.size
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            cropped = image.crop((x1, y1, x2, y2))
            cropped = cropped.resize(self.output_size, Image.Resampling.LANCZOS)
            
            crop_info: Dict[str, Any] = {
                'type': 'zoom_crop',
                'source_bbox': (x1, y1, x2, y2),
                'region_position': region.get('position', 'unknown'),
                'dominant_colors': region.get('dominant_colors', [])
            }
            
            crops.append((cropped, crop_info))
        
        return crops


# ============================================================================
# DATASET BUILDER
# ============================================================================

class Stage1DatasetBuilder:
    """
    Main class for building the Stage 1 training corpus.
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.enricher = CaptionEnricher()
        self.augmentor = ImageAugmentor()
        
        # Dataset storage
        self.pairs: List[ImageCaptionPair] = []
        self.stats = {
            'real_original': 0,
            'real_augmented': 0,
            'zoom_crops': 0,
            'synthetic': 0,
            'filtered_out': 0
        }
        
        # Optional VLM captioner
        self.vlm_captioner = None
        if self.config.USE_VLM_CAPTIONS and HAS_CUDA:
            self._init_vlm_captioner()
    
    def _init_vlm_captioner(self):
        """Initialize Qwen2-VL captioner if available"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            print("[Stage1] Loading Qwen2-VL for caption generation...")
            
            cache_dir = Path(os.path.expanduser("~/.cache/huggingface"))
            
            self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                cache_dir=str(cache_dir),
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            ).eval()
            
            self.vlm_processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                cache_dir=str(cache_dir)
            )
            
            self.process_vision_info = process_vision_info
            self.vlm_captioner = True
            print("✓ Qwen2-VL loaded for VLM captions")
            
        except Exception as e:
            print(f"⚠ Could not load Qwen2-VL: {e}")
            self.vlm_captioner = None
    
    def _generate_vlm_caption(self, image_path: str) -> str:
        """Generate caption using Qwen2-VL"""
        if not self.vlm_captioner:
            return ""
        
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Describe this aerial/satellite image in 2-3 sentences. Include scene type, main features, and spatial layout."}
                ]
            }]
            
            text = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs, _ = self.process_vision_info(messages)
            inputs = self.vlm_processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(self.vlm_model.device)
            
            with torch.no_grad():
                generated_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.85
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output = self.vlm_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output[0].strip() if output else ""
            
        except Exception as e:
            print(f"  ⚠ VLM caption error: {e}")
            return ""
    
    def _generate_image_id(self, source: str, suffix: str = "") -> str:
        """Generate unique image ID"""
        content = f"{source}_{suffix}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def process_real_image(self, image_path: Path) -> List[ImageCaptionPair]:
        """
        Process a single real image:
        1. Generate enriched caption (M2B+B2C + optional VLM)
        2. Create augmented versions
        3. Generate zoom crops
        
        Returns list of ImageCaptionPair objects
        """
        pairs = []
        image_name = image_path.stem
        
        print(f"\n  Processing: {image_name}")
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"    ✗ Could not open image: {e}")
            return pairs
        
        # === 1. Generate enriched caption ===
        caption_sources = []
        
        # M2B + B2C enrichment
        enrichment = self.enricher.enrich_caption(image)
        m2b_caption = enrichment['m2b_caption']
        caption_sources.append("m2b_b2c")
        
        # VLM caption (if available)
        vlm_caption = ""
        if self.vlm_captioner:
            vlm_caption = self._generate_vlm_caption(str(image_path))
            if vlm_caption:
                caption_sources.append("qwen2vl")
        
        # Combine captions
        if vlm_caption and m2b_caption:
            final_caption = f"{vlm_caption} {m2b_caption}"
        elif vlm_caption:
            final_caption = vlm_caption
        else:
            final_caption = m2b_caption
        
        print(f"    Caption: {final_caption[:80]}...")
        
        # === 2. Create original pair ===
        original_id = self._generate_image_id(image_name, "orig")
        original_output_path = self.config.REAL_AUG_DIR / f"{original_id}.jpg"
        
        # Save original (resized)
        resized = image.resize((512, 512), Image.Resampling.LANCZOS)
        resized.save(original_output_path, quality=95)
        
        pairs.append(ImageCaptionPair(
            image_id=original_id,
            image_path=str(original_output_path.relative_to(self.config.OUTPUT_DIR)),
            caption=final_caption,
            source=DataSource.REAL.value,
            caption_sources=caption_sources
        ))
        self.stats['real_original'] += 1
        
        # === 3. Generate augmented versions ===
        for aug_idx in range(self.config.NUM_AUGMENTATIONS_PER_IMAGE):
            aug_image, aug_params = self.augmentor.augment(image)
            aug_id = self._generate_image_id(image_name, f"aug{aug_idx}")
            aug_output_path = self.config.REAL_AUG_DIR / f"{aug_id}.jpg"
            
            aug_image.save(aug_output_path, quality=95)
            
            # Slightly modify caption for augmented versions
            aug_caption = final_caption
            if aug_params['type'] in ['horizontal_flip', 'vertical_flip']:
                aug_caption = final_caption.replace("left", "TEMP").replace("right", "left").replace("TEMP", "right")
            
            pairs.append(ImageCaptionPair(
                image_id=aug_id,
                image_path=str(aug_output_path.relative_to(self.config.OUTPUT_DIR)),
                caption=aug_caption,
                source=DataSource.REAL_AUGMENTED.value,
                source_image_id=original_id,
                caption_sources=caption_sources,
                augmentation_type=aug_params['type'],
                augmentation_params=aug_params
            ))
            self.stats['real_augmented'] += 1
        
        # === 4. Generate zoom crops ===
        if self.config.ENABLE_ZOOM_CROPS:
            regions = enrichment['regions']
            zoom_crops = self.augmentor.generate_zoom_crops(
                image, regions, self.config.ZOOM_CROP_MIN_SIZE
            )
            
            for crop_idx, (crop_img, crop_info) in enumerate(zoom_crops[:3]):  # Max 3 crops
                crop_id = self._generate_image_id(image_name, f"crop{crop_idx}")
                crop_output_path = self.config.REAL_AUG_DIR / f"{crop_id}.jpg"
                
                crop_img.save(crop_output_path, quality=95)
                
                # Generate caption for crop
                crop_enrichment = self.enricher.enrich_caption(crop_img)
                crop_caption = f"Zoomed view of {crop_info['region_position']} area. {crop_enrichment['m2b_caption']}"
                
                pairs.append(ImageCaptionPair(
                    image_id=crop_id,
                    image_path=str(crop_output_path.relative_to(self.config.OUTPUT_DIR)),
                    caption=crop_caption,
                    source=DataSource.REAL_AUGMENTED.value,
                    source_image_id=original_id,
                    caption_sources=["m2b_b2c", "zoom"],
                    augmentation_type="zoom_crop",
                    augmentation_params=crop_info
                ))
                self.stats['zoom_crops'] += 1
        
        print(f"    ✓ Generated {len(pairs)} pairs (1 orig + {self.stats['real_augmented']} aug + {self.stats['zoom_crops']} crops)")
        
        return pairs
    
    def generate_synthetic_pair(self, source_image_path: Path, 
                                 source_caption: str,
                                 variant_idx: int) -> Optional[ImageCaptionPair]:
        """
        Generate a synthetic image-caption pair using the 5-stage pipeline.
        
        Note: This is a placeholder - actual generation requires GPU and
        the full pipeline from run_hpc_pipeline.py
        """
        if not self.config.SYNTHETIC_PIPELINE_AVAILABLE:
            # Return None if pipeline not available
            # In production, this would call the actual pipeline
            return None
        
        # TODO: Integrate with actual pipeline
        # This would call:
        # 1. Real-ESRGAN upscaling
        # 2. Qwen2-VL caption enhancement
        # 3. Grounding DINO detection
        # 4. SAM precise localization
        # 5. SD3.5 generation
        
        return None
    
    def build_real_augmented_set(self) -> int:
        """
        Build the real augmented dataset from RSICD images.
        
        Returns number of pairs created.
        """
        print("\n" + "="*80)
        print("STAGE 1A: Building Real Augmented Dataset")
        print("="*80)
        
        # Find all images
        image_paths = list(self.config.RSICD_IMAGES_DIR.glob("*.jpg"))
        image_paths.extend(self.config.RSICD_IMAGES_DIR.glob("*.png"))
        
        if not image_paths:
            print(f"✗ No images found in {self.config.RSICD_IMAGES_DIR}")
            return 0
        
        print(f"Found {len(image_paths)} source images")
        
        # Process each image
        for img_path in image_paths:
            pairs = self.process_real_image(img_path)
            self.pairs.extend(pairs)
        
        print(f"\n✓ Real augmented set complete: {len(self.pairs)} pairs")
        return len(self.pairs)
    
    def build_synthetic_set(self) -> int:
        """
        Build synthetic dataset using the 5-stage pipeline.
        
        Returns number of pairs created.
        """
        print("\n" + "="*80)
        print("STAGE 1B: Building Synthetic Dataset")
        print("="*80)
        
        if not self.config.SYNTHETIC_PIPELINE_AVAILABLE:
            print("⚠ Synthetic pipeline not available (requires GPU + full pipeline)")
            print("  To enable, set SYNTHETIC_PIPELINE_AVAILABLE = True")
            print("  and ensure run_hpc_pipeline.py is configured")
            return 0
        
        # Get real images to use as seeds
        real_pairs = [p for p in self.pairs if p.source == DataSource.REAL.value]
        
        synthetic_count = 0
        for real_pair in real_pairs:
            for variant_idx in range(self.config.NUM_SYNTHETIC_PER_IMAGE):
                source_path = self.config.OUTPUT_DIR / real_pair.image_path
                synthetic_pair = self.generate_synthetic_pair(
                    source_path, real_pair.caption, variant_idx
                )
                
                if synthetic_pair:
                    self.pairs.append(synthetic_pair)
                    synthetic_count += 1
        
        self.stats['synthetic'] = synthetic_count
        print(f"\n✓ Synthetic set complete: {synthetic_count} pairs")
        return synthetic_count
    
    def save_dataset(self) -> Path:
        """
        Save the complete dataset to disk.
        
        Returns path to dataset.json
        """
        print("\n" + "="*80)
        print("Saving Dataset")
        print("="*80)
        
        # Create dataset manifest
        dataset = {
            "name": "stage1_corpus",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "stats": self.stats,
            "config": {
                "num_augmentations": self.config.NUM_AUGMENTATIONS_PER_IMAGE,
                "num_synthetic": self.config.NUM_SYNTHETIC_PER_IMAGE,
                "use_vlm": self.config.USE_VLM_CAPTIONS,
                "use_m2b_b2c": self.config.USE_M2B_B2C
            },
            "pairs": [p.to_dict() for p in self.pairs]
        }
        
        # Save main index
        dataset_path = self.config.OUTPUT_DIR / "dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Save individual metadata files
        for pair in self.pairs:
            meta_path = self.config.METADATA_DIR / f"{pair.image_id}.json"
            with open(meta_path, 'w') as f:
                json.dump(pair.to_dict(), f, indent=2)
        
        print(f"✓ Dataset saved to {dataset_path}")
        print(f"  Total pairs: {len(self.pairs)}")
        print(f"  Real original: {self.stats['real_original']}")
        print(f"  Real augmented: {self.stats['real_augmented']}")
        print(f"  Zoom crops: {self.stats['zoom_crops']}")
        print(f"  Synthetic: {self.stats['synthetic']}")
        
        return dataset_path
    
    def build(self) -> Path:
        """
        Build the complete Stage 1 dataset.
        
        Returns path to dataset.json
        """
        print("\n" + "="*80)
        print("STAGE 1: BUILD AND FREEZE AUGMENTED + SYNTHETIC DATASET")
        print("="*80)
        print(f"Output directory: {self.config.OUTPUT_DIR}")
        print(f"VLM captions: {'enabled' if self.vlm_captioner else 'disabled'}")
        print(f"M2B+B2C enrichment: {'enabled' if self.config.USE_M2B_B2C else 'disabled'}")
        
        # Build real augmented set
        self.build_real_augmented_set()
        
        # Build synthetic set
        self.build_synthetic_set()
        
        # Save everything
        return self.save_dataset()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Main entry point for Stage 1 dataset building"""
    print("\n" + "="*80)
    print("STAGE 1: DATASET BUILDER")
    print("="*80)
    
    config = DatasetConfig()
    builder = Stage1DatasetBuilder(config)
    
    dataset_path = builder.build()
    
    print("\n" + "="*80)
    print("STAGE 1 COMPLETE")
    print("="*80)
    print(f"Dataset saved to: {dataset_path}")
    print("\nNext steps:")
    print("  1. Run quality_filter.py to score and filter synthetic pairs")
    print("  2. Run stage2_lora_finetune.py for synthetic-heavy pretraining")


if __name__ == "__main__":
    main()
