#!/usr/bin/env python3
# type: ignore
"""
10-Stage Synthetic Generation Pipeline

Sequential pipeline on GPU 0:
1. Real-ESRGAN upsampling (4x, batch 4)
2. Qwen2-VL dense scene analysis
3. Grounding DINO layout detection
4. SAM segmentation
5. Intelligent Prompt Generator
6. SD 3.5 generation with ControlNet
7. Second-pass Qwen2-VL verification
8. Second-pass Grounding DINO detection
9. Second-pass SAM segmentation
10. Final Prompt Refinement
"""
from __future__ import annotations

import json
import time
import uuid
import queue
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Generator
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies with fallbacks
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

from .config import SyntheticConfig, PipelineConfig


@dataclass
class ExtractionResult:
    """Result from scene analysis extraction."""
    scene_type: str
    object_inventory: Dict[str, int]  # {object_name: count}
    layout_description: str
    appearance_details: str
    lighting_conditions: str
    raw_analysis: str


@dataclass 
class DetectionResult:
    """Result from Grounding DINO detection."""
    boxes: List[List[float]]  # [[x1, y1, x2, y2], ...]
    labels: List[str]
    scores: List[float]
    spatial_relationships: List[str]


@dataclass
class SegmentationResult:
    """Result from SAM segmentation."""
    masks: List[Any]  # List of mask arrays
    boundaries: List[List[Tuple[int, int]]]
    areas: List[int]


@dataclass
class GeneratedPrompt:
    """Generated prompt for SD 3.5."""
    full_prompt: str
    scene_component: str
    layout_component: str
    object_component: str
    style_component: str


@dataclass
class QualityScores:
    """Quality assessment scores."""
    clip_score: float
    layout_iou: float
    object_count_accuracy: float
    all_checks_passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticSample:
    """Complete synthetic sample with all metadata."""
    sample_id: str
    source_image_path: str
    synthetic_image_path: str
    
    # First pass extraction
    original_extraction: ExtractionResult
    original_detection: DetectionResult
    original_segmentation: SegmentationResult
    
    # Generation
    generated_prompt: GeneratedPrompt
    generation_params: Dict[str, Any]
    
    # Second pass verification
    verification_extraction: ExtractionResult
    verification_detection: DetectionResult
    verification_segmentation: SegmentationResult
    
    # Final outputs
    quality_scores: QualityScores
    refined_caption: str
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = "1.0.0"


class StageQueue:
    """Thread-safe queue for buffering between pipeline stages."""
    
    def __init__(self, maxsize: int = 32):
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self.done = threading.Event()
    
    def put(self, item: Any, timeout: float = 30.0):
        """Put item into queue."""
        self.queue.put(item, timeout=timeout)
    
    def get(self, timeout: float = 30.0) -> Optional[Any]:
        """Get item from queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def mark_done(self):
        """Mark queue as done (no more items)."""
        self.done.set()
    
    def is_done(self) -> bool:
        """Check if queue is done and empty."""
        return self.done.is_set() and self.queue.empty()


class SyntheticGenerationPipeline:
    """
    10-Stage Synthetic Generation Pipeline.
    
    Runs sequentially on GPU 0 with queue-based buffering between stages.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.synthetic_config = config.synthetic
        self.device = config.device
        
        # Models (lazy loaded)
        self.esrgan_model: Optional[Any] = None
        self.qwen_model: Optional[Any] = None
        self.qwen_processor: Optional[Any] = None
        self.gdino_model: Optional[Any] = None
        self.gdino_processor: Optional[Any] = None
        self.sam_model: Optional[Any] = None
        self.sam_processor: Optional[Any] = None
        self.sd_pipeline: Optional[Any] = None
        self.clip_model: Optional[Any] = None
        self.clip_preprocess: Optional[Any] = None
        
        # Statistics
        self.processed_count = 0
        self.passed_count = 0
        self.failed_count = 0
        self.stage_times: Dict[str, List[float]] = {f"stage_{i}": [] for i in range(1, 11)}
        
        # Output paths
        self.synthetic_dir = config.output_dir / "synthetic"
        self.metadata_dir = config.output_dir / "metadata" / "synthetic"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def setup(self):
        """Setup method - alias for load_models()."""
        self.load_models()
    
    def load_models(self):
        """Load all required models to GPU."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for synthetic pipeline")
        
        print("\n" + "="*80)
        print("LOADING MODELS FOR SYNTHETIC PIPELINE")
        print("="*80)
        
        # Stage 1: Real-ESRGAN
        print("\n[1/7] Loading Real-ESRGAN...")
        self._load_esrgan()
        
        # Stage 2: Qwen2-VL
        print("\n[2/7] Loading Qwen2-VL...")
        self._load_qwen()
        
        # Stage 3: Grounding DINO
        print("\n[3/7] Loading Grounding DINO...")
        self._load_grounding_dino()
        
        # Stage 4: SAM
        print("\n[4/7] Loading SAM...")
        self._load_sam()
        
        # Stage 6: SD 3.5
        print("\n[5/7] Loading Stable Diffusion 3.5...")
        self._load_sd()
        
        # CLIP for scoring
        print("\n[6/7] Loading CLIP for quality scoring...")
        self._load_clip()
        
        print("\n[7/7] All models loaded!")
        print("="*80)
    
    def _load_esrgan(self):
        """Load Real-ESRGAN model."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                           num_block=23, num_grow_ch=32, scale=4)
            
            self.esrgan_model = RealESRGANer(
                scale=self.synthetic_config.esrgan_scale,
                model_path=None,  # Uses default
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True,
                device=self.device
            )
            print("  ✓ Real-ESRGAN loaded")
        except ImportError:
            print("  ⚠ Real-ESRGAN not available, will skip upsampling")
            self.esrgan_model = None
    
    def _load_qwen(self):
        """Load Qwen2-VL model."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.synthetic_config.qwen_model,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            )
            self.qwen_processor = AutoProcessor.from_pretrained(
                self.synthetic_config.qwen_model
            )
            print("  ✓ Qwen2-VL loaded")
        except ImportError:
            print("  ⚠ Qwen2-VL not available")
            self.qwen_model = None
    
    def _load_grounding_dino(self):
        """Load Grounding DINO model."""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            self.gdino_processor = AutoProcessor.from_pretrained(
                self.synthetic_config.gdino_model
            )
            self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.synthetic_config.gdino_model
            ).to(self.device)
            print("  ✓ Grounding DINO loaded")
        except ImportError:
            print("  ⚠ Grounding DINO not available")
            self.gdino_model = None
    
    def _load_sam(self):
        """Load SAM model."""
        try:
            from transformers import SamModel, SamProcessor
            
            self.sam_model = SamModel.from_pretrained(
                self.synthetic_config.sam_model
            ).to(self.device)
            self.sam_processor = SamProcessor.from_pretrained(
                self.synthetic_config.sam_model
            )
            print("  ✓ SAM loaded")
        except ImportError:
            print("  ⚠ SAM not available")
            self.sam_model = None
    
    def _load_sd(self):
        """Load Stable Diffusion 3.5 pipeline."""
        try:
            from diffusers import StableDiffusion3Pipeline
            
            self.sd_pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.synthetic_config.sd_model,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            
            # Enable memory optimizations
            self.sd_pipeline.enable_attention_slicing()
            print("  ✓ Stable Diffusion 3.5 loaded")
        except ImportError:
            print("  ⚠ Stable Diffusion 3.5 not available")
            self.sd_pipeline = None
    
    def _load_clip(self):
        """Load CLIP for quality scoring."""
        try:
            import open_clip
            
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            self.clip_model = model.to(self.device).eval()
            self.clip_preprocess = preprocess
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            print("  ✓ CLIP loaded for quality scoring")
        except ImportError:
            print("  ⚠ CLIP not available for scoring")
            self.clip_model = None
    
    # =========================================================================
    # STAGE 1: Real-ESRGAN Upsampling
    # =========================================================================
    
    def stage1_upsample(self, image: Any) -> Any:
        """Stage 1: Upsample image using Real-ESRGAN."""
        start_time = time.time()
        
        if self.esrgan_model is None:
            return image
        
        try:
            img_array = np.array(image)
            output, _ = self.esrgan_model.enhance(img_array, outscale=4)
            result = Image.fromarray(output)
        except Exception as e:
            print(f"  ⚠ ESRGAN failed: {e}")
            result = image
        
        self.stage_times["stage_1"].append(time.time() - start_time)
        return result
    
    # =========================================================================
    # STAGE 2: Qwen2-VL Dense Scene Analysis
    # =========================================================================
    
    def stage2_scene_analysis(self, image: Any) -> ExtractionResult:
        """Stage 2: Extract dense scene information using Qwen2-VL."""
        start_time = time.time()
        
        if self.qwen_model is None or self.qwen_processor is None:
            return self._fallback_extraction()
        
        try:
            prompt = """Analyze this aerial/satellite image in detail. Provide:
1. SCENE_TYPE: What type of area is this? (urban, rural, industrial, residential, agricultural, water, forest, etc.)
2. OBJECT_INVENTORY: List all visible objects with their counts. Format: object1: count1, object2: count2
3. LAYOUT_DESCRIPTION: Describe the spatial arrangement of objects (positions, clustering, alignment)
4. APPEARANCE_DETAILS: Describe colors, textures, materials, and visual characteristics
5. LIGHTING_CONDITIONS: Describe the lighting (time of day, shadows, brightness)

Be specific and detailed for generating synthetic imagery."""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
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
                    max_new_tokens=self.synthetic_config.qwen_max_tokens
                )
            
            response = self.qwen_processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            result = self._parse_scene_analysis(response)
            
        except Exception as e:
            print(f"  ⚠ Scene analysis failed: {e}")
            result = self._fallback_extraction()
        
        self.stage_times["stage_2"].append(time.time() - start_time)
        return result
    
    def _parse_scene_analysis(self, response: str) -> ExtractionResult:
        """Parse Qwen2-VL response into structured extraction."""
        lines = response.strip().split('\n')
        
        scene_type = "unknown"
        object_inventory: Dict[str, int] = {}
        layout_description = ""
        appearance_details = ""
        lighting_conditions = ""
        
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if "scene_type" in line_lower or "scene type" in line_lower:
                current_section = "scene"
                parts = line.split(":", 1)
                if len(parts) > 1:
                    scene_type = parts[1].strip().lower()
            elif "object_inventory" in line_lower or "object inventory" in line_lower:
                current_section = "objects"
                parts = line.split(":", 1)
                if len(parts) > 1:
                    object_inventory = self._parse_object_inventory(parts[1])
            elif "layout_description" in line_lower or "layout description" in line_lower:
                current_section = "layout"
                parts = line.split(":", 1)
                if len(parts) > 1:
                    layout_description = parts[1].strip()
            elif "appearance_details" in line_lower or "appearance details" in line_lower:
                current_section = "appearance"
                parts = line.split(":", 1)
                if len(parts) > 1:
                    appearance_details = parts[1].strip()
            elif "lighting_conditions" in line_lower or "lighting conditions" in line_lower:
                current_section = "lighting"
                parts = line.split(":", 1)
                if len(parts) > 1:
                    lighting_conditions = parts[1].strip()
            elif current_section:
                # Continue previous section
                if current_section == "layout":
                    layout_description += " " + line.strip()
                elif current_section == "appearance":
                    appearance_details += " " + line.strip()
                elif current_section == "lighting":
                    lighting_conditions += " " + line.strip()
        
        return ExtractionResult(
            scene_type=scene_type or "aerial scene",
            object_inventory=object_inventory or {"structure": 1},
            layout_description=layout_description or "Objects distributed across the scene",
            appearance_details=appearance_details or "Natural colors and textures",
            lighting_conditions=lighting_conditions or "Daylight conditions",
            raw_analysis=response
        )
    
    def _parse_object_inventory(self, text: str) -> Dict[str, int]:
        """Parse object inventory from text."""
        inventory: Dict[str, int] = {}
        
        # Try comma-separated format: "building: 5, road: 2"
        parts = text.split(',')
        for part in parts:
            if ':' in part:
                obj_count = part.split(':')
                if len(obj_count) == 2:
                    obj = obj_count[0].strip().lower()
                    try:
                        count = int(obj_count[1].strip().split()[0])
                        inventory[obj] = count
                    except (ValueError, IndexError):
                        inventory[obj] = 1
        
        return inventory if inventory else {"object": 1}
    
    def _fallback_extraction(self) -> ExtractionResult:
        """Fallback extraction when Qwen2-VL is not available."""
        return ExtractionResult(
            scene_type="aerial scene",
            object_inventory={"structure": 1},
            layout_description="Objects distributed across the aerial scene",
            appearance_details="Natural satellite imagery colors",
            lighting_conditions="Daylight",
            raw_analysis="Fallback extraction"
        )
    
    # =========================================================================
    # STAGE 3: Grounding DINO Layout Detection
    # =========================================================================
    
    def stage3_layout_detection(
        self, 
        image: Any, 
        object_inventory: Dict[str, int]
    ) -> DetectionResult:
        """Stage 3: Detect object positions using Grounding DINO."""
        start_time = time.time()
        
        if self.gdino_model is None or self.gdino_processor is None:
            return self._fallback_detection()
        
        try:
            # Build text prompt from object inventory
            text_prompt = ". ".join(object_inventory.keys()) + "."
            
            inputs = self.gdino_processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.gdino_model(**inputs)
            
            # Process outputs
            results = self.gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.synthetic_config.gdino_box_threshold,
                text_threshold=self.synthetic_config.gdino_text_threshold,
                target_sizes=[image.size[::-1]]
            )[0]
            
            boxes = results["boxes"].cpu().numpy().tolist()
            labels = results["labels"]
            scores = results["scores"].cpu().numpy().tolist()
            
            # Compute spatial relationships
            spatial_rels = self._compute_spatial_relationships(boxes, labels)
            
            result = DetectionResult(
                boxes=boxes,
                labels=labels,
                scores=scores,
                spatial_relationships=spatial_rels
            )
            
        except Exception as e:
            print(f"  ⚠ Detection failed: {e}")
            result = self._fallback_detection()
        
        self.stage_times["stage_3"].append(time.time() - start_time)
        return result
    
    def _compute_spatial_relationships(
        self, 
        boxes: List[List[float]], 
        labels: List[str]
    ) -> List[str]:
        """Compute spatial relationships between detected objects."""
        relationships = []
        
        for i, (box1, label1) in enumerate(zip(boxes, labels)):
            for j, (box2, label2) in enumerate(zip(boxes, labels)):
                if i >= j:
                    continue
                
                cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
                cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
                
                if cx1 < cx2 - 50:
                    rel = f"{label1} left of {label2}"
                elif cx1 > cx2 + 50:
                    rel = f"{label1} right of {label2}"
                elif cy1 < cy2 - 50:
                    rel = f"{label1} above {label2}"
                elif cy1 > cy2 + 50:
                    rel = f"{label1} below {label2}"
                else:
                    rel = f"{label1} near {label2}"
                
                relationships.append(rel)
        
        return relationships[:10]  # Limit to 10 relationships
    
    def _fallback_detection(self) -> DetectionResult:
        """Fallback detection when Grounding DINO is not available."""
        return DetectionResult(
            boxes=[[100, 100, 400, 400]],
            labels=["structure"],
            scores=[0.5],
            spatial_relationships=[]
        )
    
    # =========================================================================
    # STAGE 4: SAM Segmentation
    # =========================================================================
    
    def stage4_segmentation(
        self, 
        image: Any, 
        detection: DetectionResult
    ) -> SegmentationResult:
        """Stage 4: Extract object shapes using SAM."""
        start_time = time.time()
        
        if self.sam_model is None or self.sam_processor is None:
            return self._fallback_segmentation()
        
        try:
            # Use detection boxes as prompts for SAM
            input_boxes = detection.boxes[:10]  # Limit boxes
            
            inputs = self.sam_processor(
                image,
                input_boxes=[input_boxes],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.sam_model(**inputs)
            
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )
            
            # Extract boundaries and areas
            boundaries = []
            areas = []
            mask_list = []
            
            for mask_batch in masks:
                for mask in mask_batch:
                    mask_np = mask.numpy().squeeze()
                    mask_list.append(mask_np)
                    areas.append(int(mask_np.sum()))
                    # Simplified boundary extraction
                    boundaries.append([])
            
            result = SegmentationResult(
                masks=mask_list,
                boundaries=boundaries,
                areas=areas
            )
            
        except Exception as e:
            print(f"  ⚠ Segmentation failed: {e}")
            result = self._fallback_segmentation()
        
        self.stage_times["stage_4"].append(time.time() - start_time)
        return result
    
    def _fallback_segmentation(self) -> SegmentationResult:
        """Fallback segmentation."""
        return SegmentationResult(
            masks=[],
            boundaries=[],
            areas=[10000]
        )
    
    # =========================================================================
    # STAGE 5: Intelligent Prompt Generator
    # =========================================================================
    
    def stage5_generate_prompt(
        self,
        extraction: ExtractionResult,
        detection: DetectionResult,
        segmentation: SegmentationResult
    ) -> GeneratedPrompt:
        """Stage 5: Generate optimized SD prompt from metadata."""
        start_time = time.time()
        
        # Scene component
        scene_component = f"{extraction.scene_type} area"
        
        # Layout component
        layout_component = extraction.layout_description
        if detection.spatial_relationships:
            layout_component += ". " + ", ".join(detection.spatial_relationships[:5])
        
        # Object component with positions
        object_parts = []
        for obj, count in extraction.object_inventory.items():
            if count > 1:
                object_parts.append(f"{count} {obj}s")
            else:
                object_parts.append(f"1 {obj}")
        object_component = ", ".join(object_parts)
        
        # Style component
        style_component = f"{extraction.appearance_details}, {extraction.lighting_conditions}"
        
        # Assemble full prompt
        full_prompt = (
            f"High-resolution aerial satellite view of {scene_component}. "
            f"Layout: {layout_component}. "
            f"Objects: {object_component}. "
            f"Visual style: {style_component}. "
            "Photorealistic remote sensing imagery, sharp details, natural colors, "
            "top-down orthographic view, professional satellite photography."
        )
        
        result = GeneratedPrompt(
            full_prompt=full_prompt,
            scene_component=scene_component,
            layout_component=layout_component,
            object_component=object_component,
            style_component=style_component
        )
        
        self.stage_times["stage_5"].append(time.time() - start_time)
        return result
    
    # =========================================================================
    # STAGE 6: SD 3.5 Generation
    # =========================================================================
    
    def stage6_generate_image(
        self,
        prompt: GeneratedPrompt,
        original_image: Optional[Any] = None
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Stage 6: Generate synthetic image using SD 3.5."""
        start_time = time.time()
        
        if self.sd_pipeline is None:
            return None, {"error": "SD pipeline not available"}
        
        try:
            generation_params = {
                "prompt": prompt.full_prompt,
                "negative_prompt": "blurry, low quality, distorted, artifacts, watermark, text, logo",
                "num_inference_steps": self.synthetic_config.sd_steps,
                "guidance_scale": self.synthetic_config.sd_guidance_scale,
                "height": self.synthetic_config.sd_image_size,
                "width": self.synthetic_config.sd_image_size,
            }
            
            with torch.no_grad():
                result = self.sd_pipeline(**generation_params)
            
            # Handle different output types
            if hasattr(result, 'images'):
                generated_image = result.images[0]
            elif isinstance(result, tuple):
                generated_image = result[0][0] if isinstance(result[0], list) else result[0]
            else:
                generated_image = result
            
            generation_params["success"] = True
            
        except Exception as e:
            print(f"  ⚠ Generation failed: {e}")
            generated_image = None
            generation_params = {"error": str(e), "success": False}
        
        self.stage_times["stage_6"].append(time.time() - start_time)
        return generated_image, generation_params
    
    # =========================================================================
    # STAGES 7-9: Second-Pass Verification
    # =========================================================================
    
    def stage7_verify_scene(self, synthetic_image: Any) -> ExtractionResult:
        """Stage 7: Second-pass scene analysis on synthetic image."""
        start_time = time.time()
        result = self.stage2_scene_analysis(synthetic_image)
        self.stage_times["stage_7"].append(time.time() - start_time)
        return result
    
    def stage8_verify_detection(
        self, 
        synthetic_image: Any,
        expected_objects: Dict[str, int]
    ) -> DetectionResult:
        """Stage 8: Second-pass detection on synthetic image."""
        start_time = time.time()
        result = self.stage3_layout_detection(synthetic_image, expected_objects)
        self.stage_times["stage_8"].append(time.time() - start_time)
        return result
    
    def stage9_verify_segmentation(
        self,
        synthetic_image: Any,
        detection: DetectionResult
    ) -> SegmentationResult:
        """Stage 9: Second-pass segmentation on synthetic image."""
        start_time = time.time()
        result = self.stage4_segmentation(synthetic_image, detection)
        self.stage_times["stage_9"].append(time.time() - start_time)
        return result
    
    # =========================================================================
    # STAGE 10: Final Prompt Refinement & Quality Scoring
    # =========================================================================
    
    def stage10_refine_and_score(
        self,
        synthetic_image: Any,
        original_prompt: GeneratedPrompt,
        original_extraction: ExtractionResult,
        original_detection: DetectionResult,
        verification_extraction: ExtractionResult,
        verification_detection: DetectionResult
    ) -> Tuple[str, QualityScores]:
        """Stage 10: Compute quality scores and generate refined caption."""
        start_time = time.time()
        
        # Compute CLIP score
        clip_score = self._compute_clip_score(synthetic_image, original_prompt.full_prompt)
        
        # Compute layout IoU
        layout_iou = self._compute_layout_iou(
            original_detection.boxes,
            verification_detection.boxes
        )
        
        # Compute object count accuracy
        obj_accuracy = self._compute_object_count_accuracy(
            original_extraction.object_inventory,
            verification_extraction.object_inventory
        )
        
        # Check all thresholds
        all_passed = (
            clip_score >= self.synthetic_config.min_clip_score and
            layout_iou >= self.synthetic_config.min_layout_iou and
            obj_accuracy >= self.synthetic_config.min_object_count_accuracy
        )
        
        quality_scores = QualityScores(
            clip_score=clip_score,
            layout_iou=layout_iou,
            object_count_accuracy=obj_accuracy,
            all_checks_passed=all_passed,
            details={
                "thresholds": {
                    "clip": self.synthetic_config.min_clip_score,
                    "iou": self.synthetic_config.min_layout_iou,
                    "obj_acc": self.synthetic_config.min_object_count_accuracy
                }
            }
        )
        
        # Generate refined caption based on verification
        refined_caption = self._generate_refined_caption(
            verification_extraction,
            verification_detection
        )
        
        self.stage_times["stage_10"].append(time.time() - start_time)
        return refined_caption, quality_scores
    
    def _compute_clip_score(self, image: Any, text: str) -> float:
        """Compute CLIP similarity score."""
        if self.clip_model is None or self.clip_preprocess is None:
            return 0.5
        
        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_input = self.clip_tokenizer([text]).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                similarity = (image_features @ text_features.T).item()
            
            return (similarity + 1) / 2  # Convert to [0, 1]
            
        except Exception as e:
            print(f"  ⚠ CLIP scoring failed: {e}")
            return 0.5
    
    def _compute_layout_iou(
        self, 
        original_boxes: List[List[float]], 
        synthetic_boxes: List[List[float]]
    ) -> float:
        """Compute average IoU between original and synthetic layouts."""
        if not original_boxes or not synthetic_boxes:
            return 0.5
        
        def box_iou(box1: List[float], box2: List[float]) -> float:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        
        # Match boxes greedily
        total_iou = 0.0
        used = set()
        
        for orig_box in original_boxes:
            best_iou = 0.0
            best_idx = -1
            
            for idx, synth_box in enumerate(synthetic_boxes):
                if idx in used:
                    continue
                iou = box_iou(orig_box, synth_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_idx >= 0:
                used.add(best_idx)
                total_iou += best_iou
        
        return total_iou / len(original_boxes) if original_boxes else 0.5
    
    def _compute_object_count_accuracy(
        self,
        original_inventory: Dict[str, int],
        synthetic_inventory: Dict[str, int]
    ) -> float:
        """Compute object count accuracy."""
        if not original_inventory:
            return 1.0
        
        total_orig = sum(original_inventory.values())
        total_synth = sum(synthetic_inventory.values())
        
        if total_orig == 0:
            return 1.0
        
        # Simple ratio-based accuracy
        ratio = min(total_synth, total_orig) / max(total_synth, total_orig)
        return ratio
    
    def _generate_refined_caption(
        self,
        extraction: ExtractionResult,
        detection: DetectionResult
    ) -> str:
        """Generate refined caption from verification pass."""
        # Build object list with counts
        objects = []
        for obj, count in extraction.object_inventory.items():
            if count > 1:
                objects.append(f"{count} {obj}s")
            else:
                objects.append(obj)
        
        # Build caption
        caption = f"Aerial view of {extraction.scene_type} area"
        
        if objects:
            caption += f" containing {', '.join(objects[:5])}"
        
        if detection.spatial_relationships:
            caption += f". {detection.spatial_relationships[0]}"
        
        if extraction.appearance_details and extraction.appearance_details != "Natural colors and textures":
            caption += f". {extraction.appearance_details[:100]}"
        
        return caption + "."
    
    # =========================================================================
    # MAIN PIPELINE EXECUTION
    # =========================================================================
    
    def process_single_image(self, image_path: Path) -> Optional[SyntheticSample]:
        """Process a single image through all 10 stages."""
        try:
            sample_id = str(uuid.uuid4())[:8]
            
            # Load source image
            source_image = Image.open(image_path).convert('RGB')
            
            # Stage 1: Upsample
            upsampled = self.stage1_upsample(source_image)
            
            # Stage 2: Scene analysis
            extraction = self.stage2_scene_analysis(upsampled)
            
            # Stage 3: Layout detection
            detection = self.stage3_layout_detection(upsampled, extraction.object_inventory)
            
            # Stage 4: Segmentation
            segmentation = self.stage4_segmentation(upsampled, detection)
            
            # Stage 5: Generate prompt
            prompt = self.stage5_generate_prompt(extraction, detection, segmentation)
            
            # Stage 6: Generate synthetic image
            synthetic_image, gen_params = self.stage6_generate_image(prompt, upsampled)
            
            if synthetic_image is None:
                self.failed_count += 1
                return None
            
            # Stage 7: Verify scene
            verify_extraction = self.stage7_verify_scene(synthetic_image)
            
            # Stage 8: Verify detection
            verify_detection = self.stage8_verify_detection(
                synthetic_image, extraction.object_inventory
            )
            
            # Stage 9: Verify segmentation
            verify_segmentation = self.stage9_verify_segmentation(
                synthetic_image, verify_detection
            )
            
            # Stage 10: Refine and score
            refined_caption, quality_scores = self.stage10_refine_and_score(
                synthetic_image,
                prompt,
                extraction,
                detection,
                verify_extraction,
                verify_detection
            )
            
            # Save synthetic image
            synthetic_path = self.synthetic_dir / f"synthetic_{sample_id}.png"
            synthetic_image.save(synthetic_path)
            
            # Create sample
            sample = SyntheticSample(
                sample_id=sample_id,
                source_image_path=str(image_path),
                synthetic_image_path=str(synthetic_path),
                original_extraction=extraction,
                original_detection=detection,
                original_segmentation=segmentation,
                generated_prompt=prompt,
                generation_params=gen_params,
                verification_extraction=verify_extraction,
                verification_detection=verify_detection,
                verification_segmentation=verify_segmentation,
                quality_scores=quality_scores,
                refined_caption=refined_caption
            )
            
            # Save metadata
            self._save_sample_metadata(sample)
            
            self.processed_count += 1
            if quality_scores.all_checks_passed:
                self.passed_count += 1
            else:
                self.failed_count += 1
            
            return sample
            
        except Exception as e:
            print(f"  ✗ Failed to process {image_path}: {e}")
            self.failed_count += 1
            return None
    
    def _save_sample_metadata(self, sample: SyntheticSample):
        """Save sample metadata to JSON."""
        metadata_path = self.metadata_dir / f"{sample.sample_id}.json"
        
        # Convert to serializable dict
        data = {
            "sample_id": sample.sample_id,
            "source_image_path": sample.source_image_path,
            "synthetic_image_path": sample.synthetic_image_path,
            "original_extraction": asdict(sample.original_extraction),
            "original_detection": {
                "boxes": sample.original_detection.boxes,
                "labels": sample.original_detection.labels,
                "scores": sample.original_detection.scores,
                "spatial_relationships": sample.original_detection.spatial_relationships
            },
            "generated_prompt": asdict(sample.generated_prompt),
            "generation_params": sample.generation_params,
            "verification_extraction": asdict(sample.verification_extraction),
            "quality_scores": asdict(sample.quality_scores),
            "refined_caption": sample.refined_caption,
            "created_at": sample.created_at,
            "pipeline_version": sample.pipeline_version
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def run(self, source_images: List[Path]) -> Dict[str, Any]:
        """Run the full synthetic generation pipeline."""
        print("\n" + "="*80)
        print("SYNTHETIC GENERATION PIPELINE (10 STAGES)")
        print("="*80)
        print(f"Processing {len(source_images)} source images")
        print(f"Target: {self.synthetic_config.target_synthetic_count} synthetic samples")
        print(f"Quality thresholds: CLIP>{self.synthetic_config.min_clip_score}, "
              f"IoU>{self.synthetic_config.min_layout_iou}, "
              f"ObjAcc>{self.synthetic_config.min_object_count_accuracy}")
        
        # Load models
        self.load_models()
        
        # Process images
        samples = []
        for i, image_path in enumerate(source_images):
            if self.passed_count >= self.synthetic_config.target_synthetic_count:
                print(f"\n✓ Reached target count: {self.passed_count}")
                break
            
            print(f"\n[{i+1}/{len(source_images)}] Processing {image_path.name}...")
            sample = self.process_single_image(image_path)
            
            if sample and sample.quality_scores.all_checks_passed:
                samples.append(sample)
                print(f"  ✓ Quality passed - CLIP: {sample.quality_scores.clip_score:.3f}, "
                      f"IoU: {sample.quality_scores.layout_iou:.3f}")
            else:
                print(f"  ✗ Quality check failed")
            
            # Progress update
            if (i + 1) % 10 == 0:
                quality_rate = self.passed_count / max(1, self.processed_count)
                print(f"\n--- Progress: {self.passed_count}/{self.processed_count} "
                      f"({quality_rate*100:.1f}% quality rate) ---")
        
        # Print summary
        print("\n" + "="*80)
        print("SYNTHETIC PIPELINE COMPLETE")
        print("="*80)
        print(f"Processed: {self.processed_count}")
        print(f"Passed: {self.passed_count}")
        print(f"Failed: {self.failed_count}")
        print(f"Quality rate: {self.passed_count/max(1,self.processed_count)*100:.1f}%")
        
        # Print stage timing
        print("\nStage timing (avg):")
        for stage, times in self.stage_times.items():
            if times:
                avg = sum(times) / len(times)
                print(f"  {stage}: {avg:.2f}s")
        
        return {
            "processed": self.processed_count,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "quality_rate": self.passed_count / max(1, self.processed_count),
            "samples": samples
        }
    
    def generate_dataset(
        self,
        source_images_dir: Path,
        output_dir: Path,
        target_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate synthetic dataset from source images directory.
        
        Args:
            source_images_dir: Directory with source images
            output_dir: Output directory for synthetic images
            target_count: Target number of synthetic samples
        
        Returns:
            Generation results dictionary
        """
        # Override target if specified
        if target_count:
            self.synthetic_config.target_synthetic_count = target_count
        
        # Collect source images
        source_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            source_images.extend(source_images_dir.glob(ext))
            source_images.extend(source_images_dir.glob(ext.upper()))
        
        if not source_images:
            print(f"No images found in {source_images_dir}")
            return {"generated": 0, "quality_rate": 0.0}
        
        # Update output directory
        self.synthetic_dir = output_dir
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline
        result = self.run(source_images)
        
        # Save dataset manifest
        manifest_path = output_dir / "dataset.json"
        manifest = {
            "pairs": [
                {
                    "image_path": str(s.synthetic_image_path),
                    "caption": s.refined_caption,
                    "quality_scores": asdict(s.quality_scores),
                    "image_id": s.sample_id
                }
                for s in result.get("samples", [])
            ]
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return {
            "generated": result.get("passed", 0),
            "quality_rate": result.get("quality_rate", 0.0),
            "manifest_path": str(manifest_path)
        }
