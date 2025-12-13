#!/usr/bin/env python3
"""
M2B (Mask-to-Box) and B2C (Box-to-Caption) Caption Enrichment
============================================================
Enriches RSICD dataset captions using pure image processing techniques.
NO ML models, NO HuggingFace dependencies.

Techniques used:
- M2B: Edge detection, contour finding, connected components → Bounding boxes
- B2C: Color analysis, texture features, shape descriptors → Region descriptions

Author: Caption Enrichment Pipeline
"""

import os
import json
import csv
from pathlib import Path
from PIL import Image, ImageFilter, ImageStat, ImageDraw
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import colorsys


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    INPUT_DIR = Path("datasets/rsicd_images")
    OUTPUT_DIR = Path("outputs/enriched_captions")
    
    # M2B Parameters
    EDGE_THRESHOLD = 30  # For edge detection sensitivity
    MIN_REGION_AREA = 500  # Minimum pixels for a valid region
    MAX_REGIONS = 15  # Maximum regions to analyze per image
    GRID_SIZE = 3  # 3x3 grid for spatial analysis
    
    # B2C Parameters  
    COLOR_BINS = 8  # Color histogram bins
    TEXTURE_WINDOW = 5  # Window size for texture analysis
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# COLOR ANALYSIS UTILITIES
# ============================================================================

# Remote sensing color vocabulary
RS_COLORS = {
    'green': {'hue_range': (60, 180), 'description': 'vegetation', 'objects': ['trees', 'grass', 'forest', 'park', 'field']},
    'dark_green': {'hue_range': (90, 150), 'sat_min': 0.3, 'val_max': 0.5, 'description': 'dense vegetation', 'objects': ['forest', 'dense trees']},
    'light_green': {'hue_range': (60, 120), 'sat_max': 0.5, 'val_min': 0.5, 'description': 'grass or lawn', 'objects': ['grass', 'lawn', 'field']},
    'blue': {'hue_range': (180, 260), 'description': 'water', 'objects': ['water', 'river', 'pond', 'lake', 'pool']},
    'dark_blue': {'hue_range': (200, 250), 'val_max': 0.4, 'description': 'deep water', 'objects': ['lake', 'river', 'ocean']},
    'gray': {'sat_max': 0.15, 'description': 'urban/roads', 'objects': ['road', 'pavement', 'concrete', 'building roof']},
    'dark_gray': {'sat_max': 0.15, 'val_max': 0.4, 'description': 'asphalt', 'objects': ['road', 'parking lot', 'asphalt']},
    'light_gray': {'sat_max': 0.15, 'val_min': 0.6, 'description': 'concrete', 'objects': ['concrete', 'sidewalk', 'building']},
    'brown': {'hue_range': (10, 40), 'description': 'bare earth', 'objects': ['bare soil', 'dirt', 'unpaved area', 'farmland']},
    'tan': {'hue_range': (30, 50), 'val_min': 0.5, 'description': 'sand or dry ground', 'objects': ['sand', 'beach', 'desert', 'dry field']},
    'red': {'hue_range': (0, 15), 'description': 'rooftops', 'objects': ['roof', 'building', 'structure']},
    'orange': {'hue_range': (15, 40), 'sat_min': 0.4, 'description': 'clay roofs', 'objects': ['clay roof', 'terracotta', 'building']},
    'white': {'val_min': 0.85, 'sat_max': 0.1, 'description': 'bright surfaces', 'objects': ['building', 'roof', 'marking', 'cloud shadow']},
    'black': {'val_max': 0.15, 'description': 'shadows/dark areas', 'objects': ['shadow', 'dark structure']},
}

# Spatial position descriptions
POSITION_NAMES = {
    (0, 0): 'top-left', (0, 1): 'top-center', (0, 2): 'top-right',
    (1, 0): 'middle-left', (1, 1): 'center', (1, 2): 'middle-right',
    (2, 0): 'bottom-left', (2, 1): 'bottom-center', (2, 2): 'bottom-right'
}


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB (0-255) to HSV (H: 0-360, S: 0-1, V: 0-1)"""
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    return h * 360, s, v


def classify_color(r: int, g: int, b: int) -> Tuple[str, str, List[str]]:
    """
    Classify an RGB color into remote sensing categories.
    Returns: (color_name, description, possible_objects)
    """
    h, s, v = rgb_to_hsv(r, g, b)
    
    # Check specific colors first (more restrictive)
    for color_name, props in RS_COLORS.items():
        match = True
        
        # Check hue range if specified
        if 'hue_range' in props:
            h_min, h_max = props['hue_range']
            if not (h_min <= h <= h_max):
                match = False
        
        # Check saturation constraints
        if 'sat_min' in props and s < props['sat_min']:
            match = False
        if 'sat_max' in props and s > props['sat_max']:
            match = False
            
        # Check value (brightness) constraints
        if 'val_min' in props and v < props['val_min']:
            match = False
        if 'val_max' in props and v > props['val_max']:
            match = False
        
        if match:
            return color_name, props['description'], props['objects']
    
    # Default fallback
    return 'mixed', 'mixed terrain', ['terrain', 'land']


# ============================================================================
# M2B: MASK TO BOX - Region Detection
# ============================================================================

class MaskToBox:
    """
    M2B: Detects regions in aerial images using image processing.
    No ML models - uses edge detection, thresholding, and contour analysis.
    """
    
    def __init__(self, config: Config):
        self.config = config
    
    def detect_edges(self, image: Image.Image) -> Image.Image:
        """Detect edges using PIL filters (Sobel-like)"""
        # Convert to grayscale
        gray = image.convert('L')
        
        # Apply edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Enhance edges
        edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        return edges
    
    def threshold_image(self, image: Image.Image, threshold: int = 128) -> np.ndarray:
        """Convert image to binary mask"""
        gray = image.convert('L')
        arr = np.array(gray)
        binary = (arr > threshold).astype(np.uint8) * 255
        return binary
    
    def find_regions_grid(self, image: Image.Image) -> List[Dict]:
        """
        Divide image into grid and analyze each cell.
        Returns list of region dictionaries with bounding boxes.
        """
        width, height = image.size
        regions = []
        
        cell_w = width // self.config.GRID_SIZE
        cell_h = height // self.config.GRID_SIZE
        
        for row in range(self.config.GRID_SIZE):
            for col in range(self.config.GRID_SIZE):
                # Calculate bounding box
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = min((col + 1) * cell_w, width)
                y2 = min((row + 1) * cell_h, height)
                
                # Crop region
                region_img = image.crop((x1, y1, x2, y2))
                
                regions.append({
                    'id': row * self.config.GRID_SIZE + col,
                    'bbox': (x1, y1, x2, y2),
                    'position': POSITION_NAMES.get((row, col), 'unknown'),
                    'grid_pos': (row, col),
                    'image': region_img,
                    'area': (x2 - x1) * (y2 - y1)
                })
        
        return regions
    
    def find_regions_adaptive(self, image: Image.Image) -> List[Dict]:
        """
        Find regions using color-based segmentation.
        Groups similar colored areas into regions.
        """
        # Quantize colors to find dominant regions
        quantized = image.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
        quantized_rgb = quantized.convert('RGB')
        
        arr = np.array(quantized_rgb)
        height, width = arr.shape[:2]
        
        # Find unique colors and their pixel locations
        pixels = arr.reshape(-1, 3)
        unique_colors, inverse, counts = np.unique(
            pixels, axis=0, return_inverse=True, return_counts=True
        )
        
        regions = []
        inverse_2d = inverse.reshape(height, width)
        
        for idx, (color, count) in enumerate(zip(unique_colors, counts)):
            if count < self.config.MIN_REGION_AREA:
                continue
            
            # Find bounding box of this color
            mask = (inverse_2d == idx)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not rows.any() or not cols.any():
                continue
                
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            
            # Determine grid position
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            grid_row = min(center_y * self.config.GRID_SIZE // height, self.config.GRID_SIZE - 1)
            grid_col = min(center_x * self.config.GRID_SIZE // width, self.config.GRID_SIZE - 1)
            
            region_img = image.crop((x1, y1, x2 + 1, y2 + 1))
            
            regions.append({
                'id': idx,
                'bbox': (int(x1), int(y1), int(x2 + 1), int(y2 + 1)),
                'position': POSITION_NAMES.get((grid_row, grid_col), 'center'),
                'grid_pos': (grid_row, grid_col),
                'image': region_img,
                'area': int(count),
                'dominant_color': tuple(color.tolist()),
                'coverage': count / (width * height)
            })
        
        # Sort by area (largest first) and limit
        regions.sort(key=lambda x: x['area'], reverse=True)
        return regions[:self.config.MAX_REGIONS]
    
    def extract_regions(self, image: Image.Image, method: str = 'hybrid') -> List[Dict]:
        """
        Main M2B function: Extract regions from image.
        
        Methods:
        - 'grid': Simple grid-based division
        - 'adaptive': Color-based segmentation  
        - 'hybrid': Combines both approaches
        """
        if method == 'grid':
            return self.find_regions_grid(image)
        elif method == 'adaptive':
            return self.find_regions_adaptive(image)
        else:  # hybrid
            grid_regions = self.find_regions_grid(image)
            adaptive_regions = self.find_regions_adaptive(image)
            
            # Merge: use grid for structure, adaptive for details
            return grid_regions + adaptive_regions[:5]


# ============================================================================
# B2C: BOX TO CAPTION - Region Description
# ============================================================================

class BoxToCaption:
    """
    B2C: Generates text descriptions for image regions.
    Uses color analysis, texture features, and spatial relationships.
    """
    
    def __init__(self, config: Config):
        self.config = config
    
    def analyze_colors(self, image: Image.Image) -> Dict:
        """Analyze color distribution in a region"""
        # Resize for faster processing
        thumb = image.copy()
        thumb.thumbnail((64, 64))
        
        pixels = list(thumb.getdata())
        
        if not pixels:
            return {'dominant': 'unknown', 'description': 'unknown area', 'objects': []}
        
        # Analyze color distribution
        color_counts = Counter()
        color_details = []
        
        for r, g, b in pixels[:1000]:  # Sample up to 1000 pixels
            color_name, desc, objects = classify_color(r, g, b)
            color_counts[color_name] += 1
            color_details.append((color_name, desc, objects))
        
        # Get dominant color
        if color_counts:
            dominant = color_counts.most_common(1)[0][0]
            dominant_info = RS_COLORS.get(dominant, {'description': 'mixed', 'objects': ['terrain']})
        else:
            dominant = 'unknown'
            dominant_info = {'description': 'unknown', 'objects': []}
        
        # Calculate color diversity
        total = sum(color_counts.values())
        diversity = len([c for c, cnt in color_counts.items() if cnt/total > 0.1])
        
        return {
            'dominant': dominant,
            'description': dominant_info.get('description', 'unknown'),
            'objects': dominant_info.get('objects', []),
            'color_counts': dict(color_counts),
            'diversity': diversity,
            'is_uniform': diversity <= 2,
            'is_mixed': diversity > 3
        }
    
    def analyze_texture(self, image: Image.Image) -> Dict:
        """Analyze texture characteristics (smoothness, patterns)"""
        gray = image.convert('L')
        
        # Resize for analysis
        thumb = gray.copy()
        thumb.thumbnail((64, 64))
        arr = np.array(thumb, dtype=np.float32)
        
        if arr.size == 0:
            return {'smoothness': 0, 'complexity': 'unknown'}
        
        # Calculate local variance (texture measure)
        local_mean = np.zeros_like(arr)
        local_var = np.zeros_like(arr)
        
        window = self.config.TEXTURE_WINDOW
        pad = window // 2
        
        # Simple variance calculation
        variance = np.var(arr)
        mean_val = np.mean(arr)
        
        # Edge density (complexity)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges.resize((64, 64)))
        edge_density = np.mean(edge_arr) / 255
        
        # Classify texture
        if variance < 500:
            smoothness = 'smooth'
            texture_type = 'uniform surface'
        elif variance < 2000:
            smoothness = 'medium'
            texture_type = 'textured surface'
        else:
            smoothness = 'rough'
            texture_type = 'complex texture'
        
        if edge_density < 0.1:
            complexity = 'simple'
        elif edge_density < 0.3:
            complexity = 'moderate'
        else:
            complexity = 'complex'
        
        return {
            'smoothness': smoothness,
            'complexity': complexity,
            'texture_type': texture_type,
            'variance': float(variance),
            'edge_density': float(edge_density),
            'brightness': float(mean_val / 255)
        }
    
    def analyze_shape(self, bbox: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> Dict:
        """Analyze shape and spatial characteristics of a region"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        img_w, img_h = image_size
        
        # Aspect ratio
        aspect = width / max(height, 1)
        
        # Relative size
        rel_area = (width * height) / (img_w * img_h)
        
        # Shape classification
        if 0.8 <= aspect <= 1.2:
            shape = 'square'
        elif aspect > 2:
            shape = 'horizontal strip'
        elif aspect < 0.5:
            shape = 'vertical strip'
        else:
            shape = 'rectangular'
        
        # Size classification
        if rel_area > 0.3:
            size = 'large'
        elif rel_area > 0.1:
            size = 'medium'
        else:
            size = 'small'
        
        return {
            'shape': shape,
            'size': size,
            'aspect_ratio': aspect,
            'relative_area': rel_area,
            'width': width,
            'height': height
        }
    
    def generate_region_caption(self, region: Dict, image_size: Tuple[int, int]) -> str:
        """Generate a natural language caption for a single region"""
        
        # Analyze the region
        color_info = self.analyze_colors(region['image'])
        texture_info = self.analyze_texture(region['image'])
        shape_info = self.analyze_shape(region['bbox'], image_size)
        
        # Build caption components
        parts = []
        
        # Position
        position = region.get('position', 'center')
        
        # Main object/feature
        objects = color_info.get('objects', ['area'])
        main_object = objects[0] if objects else 'area'
        
        # Size modifier
        size = shape_info.get('size', 'medium')
        
        # Texture modifier
        texture = texture_info.get('smoothness', 'textured')
        
        # Build description
        if color_info.get('is_uniform', False):
            desc = f"{size} {color_info['description']} area"
        else:
            desc = f"{size} {texture} {main_object}"
        
        # Add position if not center
        if position != 'center':
            caption = f"{desc} in the {position}"
        else:
            caption = f"{desc} in the center"
        
        # Store analysis for reference
        region['color_analysis'] = color_info
        region['texture_analysis'] = texture_info
        region['shape_analysis'] = shape_info
        region['caption'] = caption
        
        return caption
    
    def describe_regions(self, regions: List[Dict], image_size: Tuple[int, int]) -> List[str]:
        """Generate captions for all regions"""
        captions = []
        for region in regions:
            caption = self.generate_region_caption(region, image_size)
            captions.append(caption)
        return captions


# ============================================================================
# CAPTION ENRICHMENT ENGINE
# ============================================================================

class CaptionEnricher:
    """
    Main class that combines M2B and B2C to enrich captions.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.m2b = MaskToBox(self.config)
        self.b2c = BoxToCaption(self.config)
    
    def enrich_caption(self, image_path: str, original_caption: str = "") -> Dict:
        """
        Enrich a caption for a single image.
        
        Returns dict with:
        - original_caption: Input caption
        - enriched_caption: Enhanced caption with region details
        - regions: List of detected regions with their descriptions
        - scene_analysis: Overall scene analysis
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_size = image.size
        
        # M2B: Extract regions
        regions = self.m2b.extract_regions(image, method='hybrid')
        
        # B2C: Generate region captions
        region_captions = self.b2c.describe_regions(regions, image_size)
        
        # Analyze overall scene
        scene_analysis = self._analyze_scene(regions, image)
        
        # Combine into enriched caption
        enriched = self._compose_enriched_caption(
            original_caption, region_captions, scene_analysis
        )
        
        return {
            'image_path': str(image_path),
            'original_caption': original_caption,
            'enriched_caption': enriched,
            'region_count': len(regions),
            'regions': [
                {
                    'id': r['id'],
                    'position': r['position'],
                    'bbox': r['bbox'],
                    'caption': r.get('caption', ''),
                    'color': r.get('color_analysis', {}).get('dominant', 'unknown'),
                    'area': r.get('area', 0)
                }
                for r in regions
            ],
            'scene_analysis': scene_analysis
        }
    
    def _analyze_scene(self, regions: List[Dict], image: Image.Image) -> Dict:
        """Analyze overall scene characteristics"""
        # Collect all color analyses
        all_colors = Counter()
        all_objects = []
        
        for region in regions:
            if 'color_analysis' in region:
                for color, count in region['color_analysis'].get('color_counts', {}).items():
                    all_colors[color] += count
                all_objects.extend(region['color_analysis'].get('objects', []))
        
        # Determine scene type
        green_pct = (all_colors.get('green', 0) + all_colors.get('dark_green', 0) + 
                    all_colors.get('light_green', 0))
        gray_pct = (all_colors.get('gray', 0) + all_colors.get('dark_gray', 0) + 
                   all_colors.get('light_gray', 0))
        blue_pct = all_colors.get('blue', 0) + all_colors.get('dark_blue', 0)
        brown_pct = all_colors.get('brown', 0) + all_colors.get('tan', 0)
        
        total = sum(all_colors.values()) or 1
        
        # Classify scene
        if green_pct / total > 0.5:
            scene_type = 'vegetation-dominant'
            scene_desc = 'area with abundant vegetation'
        elif gray_pct / total > 0.4:
            scene_type = 'urban'
            scene_desc = 'urban or developed area'
        elif blue_pct / total > 0.3:
            scene_type = 'water'
            scene_desc = 'area with water bodies'
        elif brown_pct / total > 0.4:
            scene_type = 'agricultural'
            scene_desc = 'agricultural or bare land area'
        else:
            scene_type = 'mixed'
            scene_desc = 'mixed land use area'
        
        # Get unique objects
        unique_objects = list(set(all_objects))[:10]
        
        return {
            'scene_type': scene_type,
            'scene_description': scene_desc,
            'dominant_colors': dict(all_colors.most_common(5)),
            'detected_features': unique_objects,
            'color_distribution': {
                'vegetation': green_pct / total,
                'urban': gray_pct / total,
                'water': blue_pct / total,
                'bare_land': brown_pct / total
            }
        }
    
    def _compose_enriched_caption(self, original: str, region_captions: List[str], 
                                   scene_analysis: Dict) -> str:
        """Compose the final enriched caption"""
        parts = []
        
        # Start with scene description
        scene_desc = scene_analysis.get('scene_description', 'aerial view')
        parts.append(f"This aerial image shows a {scene_desc}.")
        
        # Add original caption if provided
        if original and original.strip():
            # Clean up original
            orig_clean = original.strip()
            if not orig_clean.endswith('.'):
                orig_clean += '.'
            parts.append(orig_clean)
        
        # Group regions by position for coherent description
        position_groups = {}
        for caption in region_captions[:9]:  # Limit to grid regions
            # Extract position from caption
            for pos in POSITION_NAMES.values():
                if pos in caption:
                    if pos not in position_groups:
                        position_groups[pos] = []
                    position_groups[pos].append(caption)
                    break
        
        # Add notable features
        features = scene_analysis.get('detected_features', [])[:5]
        if features:
            feature_str = ', '.join(features[:-1])
            if len(features) > 1:
                feature_str += f' and {features[-1]}'
            else:
                feature_str = features[0]
            parts.append(f"Notable features include {feature_str}.")
        
        # Add spatial descriptions (select most informative)
        spatial_descs = []
        for pos, captions in list(position_groups.items())[:3]:
            if captions:
                spatial_descs.append(captions[0])
        
        if spatial_descs:
            parts.append(' '.join(spatial_descs[:2]))
        
        return ' '.join(parts)
    
    def process_dataset(self, image_dir: str = None, output_file: str = None,
                        original_captions: Dict[str, str] = None) -> List[Dict]:
        """
        Process entire dataset and enrich captions.
        
        Args:
            image_dir: Directory containing images
            output_file: Output JSON file path
            original_captions: Dict mapping image filename to original caption
        """
        image_dir = Path(image_dir or self.config.INPUT_DIR)
        output_file = output_file or str(self.config.OUTPUT_DIR / "enriched_captions.json")
        original_captions = original_captions or {}
        
        results = []
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        images = [f for f in image_dir.iterdir() 
                  if f.suffix.lower() in image_extensions]
        
        print(f"\n{'='*60}")
        print("M2B + B2C Caption Enrichment Pipeline")
        print(f"{'='*60}")
        print(f"Input directory: {image_dir}")
        print(f"Found {len(images)} images")
        print(f"Output: {output_file}")
        print(f"{'='*60}\n")
        
        for idx, img_path in enumerate(images):
            print(f"[{idx+1}/{len(images)}] Processing: {img_path.name}")
            
            try:
                # Get original caption if available
                orig_caption = original_captions.get(img_path.name, "")
                
                # Enrich caption
                result = self.enrich_caption(str(img_path), orig_caption)
                results.append(result)
                
                print(f"  ✓ Enriched: {result['enriched_caption'][:80]}...")
                print(f"  ✓ Detected {result['region_count']} regions")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    'image_path': str(img_path),
                    'error': str(e)
                })
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Processed {len(results)} images")
        print(f"✓ Results saved to: {output_file}")
        print(f"{'='*60}")
        
        return results


# ============================================================================
# VISUALIZATION (Optional)
# ============================================================================

def visualize_regions(image_path: str, regions: List[Dict], output_path: str = None):
    """Draw detected regions on image for debugging"""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    for idx, region in enumerate(regions[:len(colors)]):
        bbox = region['bbox']
        color = colors[idx % len(colors)]
        draw.rectangle(bbox, outline=color, width=2)
        
        # Add label
        label = f"{region.get('position', idx)}"
        draw.text((bbox[0], bbox[1] - 15), label, fill=color)
    
    if output_path:
        image.save(output_path)
        print(f"Visualization saved to: {output_path}")
    
    return image


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for caption enrichment"""
    import argparse
    
    parser = argparse.ArgumentParser(description='M2B + B2C Caption Enrichment')
    parser.add_argument('--input', '-i', default='datasets/rsicd_images',
                        help='Input image directory')
    parser.add_argument('--output', '-o', default='outputs/enriched_captions/results.json',
                        help='Output JSON file')
    parser.add_argument('--single', '-s', default=None,
                        help='Process single image')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate visualization images')
    parser.add_argument('--max-images', '-m', type=int, default=None,
                        help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    config = Config()
    enricher = CaptionEnricher(config)
    
    if args.single:
        # Process single image
        result = enricher.enrich_caption(args.single)
        print(f"\nOriginal: {result['original_caption']}")
        print(f"\nEnriched: {result['enriched_caption']}")
        print(f"\nRegions detected: {result['region_count']}")
        print(f"Scene type: {result['scene_analysis']['scene_type']}")
        
        if args.visualize:
            vis_path = args.single.replace('.', '_regions.')
            visualize_regions(args.single, result['regions'], vis_path)
    else:
        # Process dataset
        enricher.process_dataset(
            image_dir=args.input,
            output_file=args.output
        )


if __name__ == "__main__":
    main()
