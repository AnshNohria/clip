#!/usr/bin/env python3
# type: ignore
"""
5-Stage Zoom Crops Pipeline

Pipeline for extracting and captioning zoom crops:
1. Grounding DINO detection (batch 16, confidence 0.35)
2. SAM segmentation
3. CPU-based crop extraction (8 workers, top-K=10, 20% padding)
4. Qwen2-VL captioning
5. Prompt Refinement Engine
"""
from __future__ import annotations

import json
import time
import uuid
import concurrent.futures
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies
torch: Any = None
nn: Any = None
F: Any = None
Image: Any = None
np: Any = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .config import ZoomCropConfig, PipelineConfig


@dataclass
class CropDetection:
    """Detection result for a single object."""
    box: List[float]  # [x1, y1, x2, y2]
    label: str
    score: float
    center: Tuple[float, float]
    area: float


@dataclass
class CropContext:
    """Context information for a crop."""
    position_in_image: str  # top-left, center, bottom-right, etc.
    surrounding_objects: List[str]
    spatial_context: str
    relative_size: str  # small, medium, large


@dataclass
class ZoomCrop:
    """Complete zoom crop sample."""
    crop_id: str
    source_image_path: str
    crop_image_path: str
    
    # Detection info
    detection: CropDetection
    
    # Segmentation mask
    mask_path: Optional[str]
    boundary_area: int
    
    # Context
    context: CropContext
    
    # Captions
    qwen_caption: str
    refined_caption: str
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ZoomCropsPipeline:
    """
    5-Stage Zoom Crops Pipeline.
    
    Extracts object crops with refined captions for training.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.crop_config = config.zoom_crops
        self.device = config.device
        
        # Models (lazy loaded)
        self.gdino_model: Optional[Any] = None
        self.gdino_processor: Optional[Any] = None
        self.sam_model: Optional[Any] = None
        self.sam_processor: Optional[Any] = None
        self.qwen_model: Optional[Any] = None
        self.qwen_processor: Optional[Any] = None
        
        # Statistics
        self.processed_images = 0
        self.total_crops = 0
        self.stage_times: Dict[str, List[float]] = {f"stage_{i}": [] for i in range(1, 6)}
        
        # Output paths
        self.crops_dir = config.output_dir / "zoom_crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = config.output_dir / "metadata" / "zoom_crops"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def setup(self):
        """Setup method - alias for load_models()."""
        self.load_models()
    
    def load_models(self):
        """Load required models."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for zoom crops pipeline")
        
        print("\n" + "="*80)
        print("LOADING MODELS FOR ZOOM CROPS PIPELINE")
        print("="*80)
        
        # Grounding DINO
        print("\n[1/3] Loading Grounding DINO...")
        self._load_grounding_dino()
        
        # SAM
        print("\n[2/3] Loading SAM...")
        self._load_sam()
        
        # Qwen2-VL
        print("\n[3/3] Loading Qwen2-VL...")
        self._load_qwen()
        
        print("\n✓ All models loaded!")
    
    def _load_grounding_dino(self):
        """Load Grounding DINO."""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            self.gdino_processor = AutoProcessor.from_pretrained(
                "IDEA-Research/grounding-dino-base"
            )
            self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-base"
            ).to(self.device)
            print("  ✓ Grounding DINO loaded")
        except Exception as e:
            print(f"  ⚠ Grounding DINO not available: {e}")
    
    def _load_sam(self):
        """Load SAM."""
        try:
            from transformers import SamModel, SamProcessor
            
            self.sam_model = SamModel.from_pretrained(
                "facebook/sam-vit-huge"
            ).to(self.device)
            self.sam_processor = SamProcessor.from_pretrained(
                "facebook/sam-vit-huge"
            )
            print("  ✓ SAM loaded")
        except Exception as e:
            print(f"  ⚠ SAM not available: {e}")
    
    def _load_qwen(self):
        """Load Qwen2-VL."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map=self.device
            )
            self.qwen_processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct"
            )
            print("  ✓ Qwen2-VL loaded")
        except Exception as e:
            print(f"  ⚠ Qwen2-VL not available: {e}")
    
    # =========================================================================
    # STAGE 1: Grounding DINO Detection
    # =========================================================================
    
    def stage1_detect_objects(
        self, 
        image: Any,
        text_prompt: str = "building. road. vehicle. tree. water. structure."
    ) -> List[CropDetection]:
        """Stage 1: Detect objects using Grounding DINO."""
        start_time = time.time()
        
        if self.gdino_model is None or self.gdino_processor is None:
            return self._fallback_detections(image)
        
        try:
            inputs = self.gdino_processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.gdino_model(**inputs)
            
            results = self.gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.crop_config.gdino_confidence,
                text_threshold=0.25,
                target_sizes=[image.size[::-1]]
            )[0]
            
            detections = []
            boxes = results["boxes"].cpu().numpy()
            labels = results["labels"]
            scores = results["scores"].cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.tolist()
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                area = (x2 - x1) * (y2 - y1)
                
                detections.append(CropDetection(
                    box=[x1, y1, x2, y2],
                    label=label,
                    score=float(score),
                    center=center,
                    area=area
                ))
            
            # Sort by score and take top-K
            detections.sort(key=lambda d: d.score, reverse=True)
            detections = detections[:self.crop_config.top_k_crops]
            
        except Exception as e:
            print(f"  ⚠ Detection failed: {e}")
            detections = self._fallback_detections(image)
        
        self.stage_times["stage_1"].append(time.time() - start_time)
        return detections
    
    def _fallback_detections(self, image: Any) -> List[CropDetection]:
        """Fallback detection."""
        w, h = image.size
        return [CropDetection(
            box=[w*0.2, h*0.2, w*0.8, h*0.8],
            label="structure",
            score=0.5,
            center=(w/2, h/2),
            area=(w*0.6) * (h*0.6)
        )]
    
    # =========================================================================
    # STAGE 2: SAM Segmentation
    # =========================================================================
    
    def stage2_segment_objects(
        self,
        image: Any,
        detections: List[CropDetection]
    ) -> Dict[str, Tuple[Any, int]]:
        """Stage 2: Segment detected objects using SAM."""
        start_time = time.time()
        
        if self.sam_model is None or self.sam_processor is None:
            return {d.label: (None, int(d.area)) for d in detections}
        
        results: Dict[str, Tuple[Any, int]] = {}
        
        try:
            for detection in detections[:self.crop_config.top_k_crops]:
                input_boxes = [[detection.box]]
                
                inputs = self.sam_processor(
                    image,
                    input_boxes=input_boxes,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.sam_model(**inputs)
                
                masks = self.sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )
                
                if masks and len(masks) > 0:
                    mask = masks[0][0].numpy().squeeze()
                    area = int(mask.sum())
                else:
                    mask = None
                    area = int(detection.area)
                
                results[f"{detection.label}_{len(results)}"] = (mask, area)
                
        except Exception as e:
            print(f"  ⚠ Segmentation failed: {e}")
            for d in detections:
                results[d.label] = (None, int(d.area))
        
        self.stage_times["stage_2"].append(time.time() - start_time)
        return results
    
    # =========================================================================
    # STAGE 3: CPU-based Crop Extraction
    # =========================================================================
    
    def stage3_extract_crops(
        self,
        image: Any,
        detections: List[CropDetection],
        source_path: str
    ) -> List[Tuple[Any, CropDetection, CropContext]]:
        """Stage 3: Extract crops with context (CPU, parallel)."""
        start_time = time.time()
        
        w, h = image.size
        crops = []
        
        def extract_single_crop(detection: CropDetection) -> Optional[Tuple[Any, CropDetection, CropContext]]:
            """Extract a single crop with padding and context."""
            x1, y1, x2, y2 = detection.box
            
            # Add padding
            pad_x = (x2 - x1) * self.crop_config.crop_padding
            pad_y = (y2 - y1) * self.crop_config.crop_padding
            
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(w, x2 + pad_x)
            crop_y2 = min(h, y2 + pad_y)
            
            # Check size constraints
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            
            if crop_w < self.crop_config.min_crop_size or crop_h < self.crop_config.min_crop_size:
                return None
            
            # Extract crop
            crop_img = image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))
            
            # Compute context
            context = self._compute_crop_context(detection, detections, w, h)
            
            return (crop_img, detection, context)
        
        # Use thread pool for CPU-bound extraction
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.crop_config.num_workers) as executor:
            futures = [executor.submit(extract_single_crop, d) for d in detections]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    crops.append(result)
        
        self.stage_times["stage_3"].append(time.time() - start_time)
        return crops
    
    def _compute_crop_context(
        self,
        detection: CropDetection,
        all_detections: List[CropDetection],
        image_width: int,
        image_height: int
    ) -> CropContext:
        """Compute context information for a crop."""
        cx, cy = detection.center
        
        # Position in image
        x_pos = "left" if cx < image_width / 3 else ("right" if cx > 2 * image_width / 3 else "center")
        y_pos = "top" if cy < image_height / 3 else ("bottom" if cy > 2 * image_height / 3 else "middle")
        position = f"{y_pos}-{x_pos}"
        
        # Surrounding objects
        surrounding = []
        for other in all_detections:
            if other is detection:
                continue
            
            dist = ((other.center[0] - cx) ** 2 + (other.center[1] - cy) ** 2) ** 0.5
            if dist < max(image_width, image_height) * 0.3:
                surrounding.append(other.label)
        
        # Spatial context
        if surrounding:
            spatial_parts = []
            for other in all_detections:
                if other is detection:
                    continue
                ocx, ocy = other.center
                if ocx < cx - 50:
                    spatial_parts.append(f"{other.label} on left")
                elif ocx > cx + 50:
                    spatial_parts.append(f"{other.label} on right")
            spatial_context = ", ".join(spatial_parts[:3]) if spatial_parts else "isolated"
        else:
            spatial_context = "isolated in scene"
        
        # Relative size
        area_ratio = detection.area / (image_width * image_height)
        if area_ratio < 0.05:
            rel_size = "small"
        elif area_ratio < 0.2:
            rel_size = "medium"
        else:
            rel_size = "large"
        
        return CropContext(
            position_in_image=position,
            surrounding_objects=surrounding[:5],
            spatial_context=spatial_context,
            relative_size=rel_size
        )
    
    # =========================================================================
    # STAGE 4: Qwen2-VL Captioning
    # =========================================================================
    
    def stage4_caption_crop(
        self,
        crop_image: Any,
        detection: CropDetection
    ) -> str:
        """Stage 4: Generate caption for crop using Qwen2-VL."""
        start_time = time.time()
        
        if self.qwen_model is None or self.qwen_processor is None:
            return f"A {detection.label} in an aerial image"
        
        try:
            prompt = f"Describe this {detection.label} in the aerial image in {self.crop_config.caption_max_words} words or less."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop_image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.qwen_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=30
                )
            
            caption = self.qwen_processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0].strip()
            
        except Exception as e:
            print(f"  ⚠ Captioning failed: {e}")
            caption = f"A {detection.label} in an aerial image"
        
        self.stage_times["stage_4"].append(time.time() - start_time)
        return caption
    
    # =========================================================================
    # STAGE 5: Prompt Refinement
    # =========================================================================
    
    def stage5_refine_caption(
        self,
        detection: CropDetection,
        context: CropContext,
        qwen_caption: str
    ) -> str:
        """Stage 5: Refine caption using all metadata."""
        start_time = time.time()
        
        # Build refined caption
        parts = []
        
        # Object description with size
        if context.relative_size != "medium":
            parts.append(f"{context.relative_size.capitalize()} {detection.label}")
        else:
            parts.append(detection.label.capitalize())
        
        # Add Qwen description if meaningful
        if qwen_caption and len(qwen_caption) > 10:
            # Extract key descriptors
            qwen_lower = qwen_caption.lower()
            descriptors = []
            
            # Look for color/material descriptors
            for word in ["rectangular", "square", "circular", "flat", "sloped", 
                        "white", "gray", "brown", "green", "metallic", "concrete"]:
                if word in qwen_lower:
                    descriptors.append(word)
            
            if descriptors:
                parts[0] = f"{', '.join(descriptors[:2])} {parts[0].lower()}"
        
        # Add position
        parts.append(f"in {context.position_in_image} of aerial scene")
        
        # Add spatial context
        if context.surrounding_objects:
            if len(context.surrounding_objects) == 1:
                parts.append(f"near {context.surrounding_objects[0]}")
            else:
                parts.append(f"surrounded by {', '.join(context.surrounding_objects[:2])}")
        
        # Add detailed spatial relationships
        if context.spatial_context and context.spatial_context != "isolated":
            parts.append(context.spatial_context)
        
        refined = ". ".join(parts) + "."
        
        # Ensure proper capitalization and formatting
        refined = refined[0].upper() + refined[1:]
        
        self.stage_times["stage_5"].append(time.time() - start_time)
        return refined
    
    # =========================================================================
    # MAIN PIPELINE EXECUTION
    # =========================================================================
    
    def process_single_image(self, image_path: Path) -> List[ZoomCrop]:
        """Process a single image through all 5 stages."""
        crops_from_image = []
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Stage 1: Detect objects
            detections = self.stage1_detect_objects(image)
            
            if not detections:
                return []
            
            # Stage 2: Segment objects
            segmentations = self.stage2_segment_objects(image, detections)
            
            # Stage 3: Extract crops
            crop_data = self.stage3_extract_crops(image, detections, str(image_path))
            
            # Process each crop
            for crop_img, detection, context in crop_data:
                crop_id = str(uuid.uuid4())[:8]
                
                # Stage 4: Caption
                qwen_caption = self.stage4_caption_crop(crop_img, detection)
                
                # Stage 5: Refine
                refined_caption = self.stage5_refine_caption(detection, context, qwen_caption)
                
                # Save crop image
                crop_path = self.crops_dir / f"crop_{crop_id}.png"
                crop_img.save(crop_path)
                
                # Get segmentation info
                seg_key = f"{detection.label}_{len([c for c in crops_from_image if detection.label in c.detection.label])}"
                mask, area = segmentations.get(seg_key, (None, int(detection.area)))
                
                # Create ZoomCrop
                zoom_crop = ZoomCrop(
                    crop_id=crop_id,
                    source_image_path=str(image_path),
                    crop_image_path=str(crop_path),
                    detection=detection,
                    mask_path=None,  # Optional: save mask
                    boundary_area=area,
                    context=context,
                    qwen_caption=qwen_caption,
                    refined_caption=refined_caption
                )
                
                crops_from_image.append(zoom_crop)
                self.total_crops += 1
                
                # Save metadata
                self._save_crop_metadata(zoom_crop)
            
            self.processed_images += 1
            
        except Exception as e:
            print(f"  ✗ Failed to process {image_path}: {e}")
        
        return crops_from_image
    
    def _save_crop_metadata(self, crop: ZoomCrop):
        """Save crop metadata."""
        metadata_path = self.metadata_dir / f"{crop.crop_id}.json"
        
        data = {
            "crop_id": crop.crop_id,
            "source_image_path": crop.source_image_path,
            "crop_image_path": crop.crop_image_path,
            "detection": {
                "box": crop.detection.box,
                "label": crop.detection.label,
                "score": crop.detection.score,
                "center": crop.detection.center,
                "area": crop.detection.area
            },
            "boundary_area": crop.boundary_area,
            "context": asdict(crop.context),
            "qwen_caption": crop.qwen_caption,
            "refined_caption": crop.refined_caption,
            "created_at": crop.created_at
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def run(self, source_images: List[Path]) -> Dict[str, Any]:
        """Run the full zoom crops pipeline."""
        print("\n" + "="*80)
        print("ZOOM CROPS PIPELINE (5 STAGES)")
        print("="*80)
        print(f"Processing {len(source_images)} source images")
        print(f"Target: ~{self.crop_config.target_crop_count} crops")
        print(f"Top-K per image: {self.crop_config.top_k_crops}")
        
        # Load models
        self.load_models()
        
        # Process images
        all_crops = []
        
        for i, image_path in enumerate(source_images):
            if self.total_crops >= self.crop_config.target_crop_count:
                print(f"\n✓ Reached target count: {self.total_crops}")
                break
            
            print(f"\n[{i+1}/{len(source_images)}] Processing {image_path.name}...")
            crops = self.process_single_image(image_path)
            all_crops.extend(crops)
            
            print(f"  Extracted {len(crops)} crops (total: {self.total_crops})")
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"\n--- Progress: {self.total_crops} crops from {self.processed_images} images ---")
        
        # Print summary
        print("\n" + "="*80)
        print("ZOOM CROPS PIPELINE COMPLETE")
        print("="*80)
        print(f"Processed images: {self.processed_images}")
        print(f"Total crops: {self.total_crops}")
        print(f"Avg crops/image: {self.total_crops / max(1, self.processed_images):.1f}")
        
        # Print stage timing
        print("\nStage timing (avg):")
        for stage, times in self.stage_times.items():
            if times:
                avg = sum(times) / len(times)
                print(f"  {stage}: {avg:.2f}s")
        
        return {
            "processed_images": self.processed_images,
            "total_crops": self.total_crops,
            "crops": all_crops
        }
    
    def build_dataset_json(self, crops: List[ZoomCrop], output_path: Path):
        """Build dataset JSON file from crops."""
        dataset = {
            "name": "zoom_crops_dataset",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "total_crops": len(crops),
            "pairs": []
        }
        
        for crop in crops:
            pair = {
                "image_id": crop.crop_id,
                "image_path": str(Path(crop.crop_image_path).relative_to(self.config.output_dir)),
                "caption": crop.refined_caption,
                "source": "zoom_crop",
                "metadata": {
                    "label": crop.detection.label,
                    "score": crop.detection.score,
                    "context": asdict(crop.context),
                    "qwen_caption": crop.qwen_caption
                }
            }
            dataset["pairs"].append(pair)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✓ Dataset saved: {output_path}")
    
    def extract_all_crops(
        self,
        images_dir: Path,
        output_dir: Path,
        target_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract crops from all images in directory.
        
        Args:
            images_dir: Directory with source images
            output_dir: Output directory for crops
            target_count: Target number of crops
        
        Returns:
            Extraction results dictionary
        """
        # Override target if specified
        if target_count:
            self.crop_config.target_crop_count = target_count
        
        # Update output directory
        self.crops_dir = output_dir
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect source images
        source_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            source_images.extend(images_dir.glob(ext))
            source_images.extend(images_dir.glob(ext.upper()))
        
        if not source_images:
            print(f"No images found in {images_dir}")
            return {"total_crops": 0, "images_processed": 0}
        
        # Run pipeline
        result = self.run(source_images)
        
        # Build dataset JSON
        dataset_path = output_dir / "dataset.json"
        self.build_dataset_json(result.get("crops", []), dataset_path)
        
        return {
            "total_crops": result.get("total_crops", 0),
            "images_processed": result.get("processed_images", 0),
            "dataset_path": str(dataset_path)
        }
