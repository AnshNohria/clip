#!/usr/bin/env python3
# type: ignore
"""
Synthetic Generation Pipeline for Remote Sensing Images

Simplified 9-stage pipeline (Real-ESRGAN removed):
1. (Skipped) - Real-ESRGAN removed
2. Qwen2-VL dense scene analysis  
3. Grounding DINO layout detection
4. SAM segmentation
5. Intelligent Prompt Generator
6. Stable Diffusion generation
7. Second-pass Qwen2-VL verification
8. Second-pass Grounding DINO detection
9. Second-pass SAM segmentation
10. Final Prompt Refinement & Quality Scoring

All models run sequentially on GPU with lazy loading to manage memory.
"""
from __future__ import annotations

import gc
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from PIL import Image

from .config import PipelineConfig


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractionResult:
    """Result from scene analysis extraction."""
    scene_type: str
    object_inventory: Dict[str, int]
    layout_description: str
    appearance_details: str
    lighting_conditions: str
    raw_analysis: str


@dataclass
class DetectionResult:
    """Result from Grounding DINO detection."""
    boxes: List[List[float]]
    labels: List[str]
    scores: List[float]
    spatial_relationships: List[str]


@dataclass
class SegmentationResult:
    """Result from SAM segmentation."""
    masks: List[Any]
    boundaries: List[List[Tuple[int, int]]]
    areas: List[int]


@dataclass
class GeneratedPrompt:
    """Generated prompt for Stable Diffusion."""
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
    original_extraction: ExtractionResult
    original_detection: DetectionResult
    original_segmentation: SegmentationResult
    generated_prompt: GeneratedPrompt
    generation_params: Dict[str, Any]
    verification_extraction: ExtractionResult
    verification_detection: DetectionResult
    verification_segmentation: SegmentationResult
    quality_scores: QualityScores
    refined_caption: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = "2.0.0"


# =============================================================================
# Main Pipeline Class
# =============================================================================

class SyntheticGenerationPipeline:
    """
    Synthetic Generation Pipeline with lazy model loading.
    
    Models are loaded one at a time to manage GPU memory efficiently.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.synthetic_config = config.synthetic
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Model references (all start as None - lazy loaded)
        self.qwen_model = None
        self.qwen_processor = None
        self.gdino_model = None
        self.gdino_processor = None
        self.sam_model = None
        self.sam_processor = None
        self.sd_pipeline = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        
        # Statistics
        self.processed_count = 0
        self.passed_count = 0
        self.failed_count = 0
        self.stage_times: Dict[str, List[float]] = {f"stage_{i}": [] for i in range(2, 11)}
        
        # Output paths
        self.output_root = config.output_dir.parent
        self.synthetic_dir = self.output_root / "images"
        self.metadata_dir = self.output_root / "metadata"
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # GPU Memory Management
    # =========================================================================
    
    def _clear_gpu_memory(self):
        """Aggressively clear GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _get_free_memory(self) -> float:
        """Get free GPU memory in GB."""
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return free / (1024**3)
        return 0.0
    
    def _unload_all_models(self):
        """Unload all models from GPU."""
        if self.qwen_model is not None:
            del self.qwen_model
            del self.qwen_processor
            self.qwen_model = None
            self.qwen_processor = None
        
        if self.gdino_model is not None:
            del self.gdino_model
            del self.gdino_processor
            self.gdino_model = None
            self.gdino_processor = None
        
        if self.sam_model is not None:
            del self.sam_model
            del self.sam_processor
            self.sam_model = None
            self.sam_processor = None
        
        if self.sd_pipeline is not None:
            del self.sd_pipeline
            self.sd_pipeline = None
        
        if self.clip_model is not None:
            del self.clip_model
            del self.clip_preprocess
            del self.clip_tokenizer
            self.clip_model = None
            self.clip_preprocess = None
            self.clip_tokenizer = None
        
        self._clear_gpu_memory()
    
    # =========================================================================
    # Model Loading (Lazy)
    # =========================================================================
    
    def _load_qwen(self):
        """Load Qwen2-VL model."""
        if self.qwen_model is not None:
            return
        
        self._unload_all_models()
        print(f"  Loading Qwen2-VL... (Free: {self._get_free_memory():.1f}GB)")
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            self.qwen_processor = AutoProcessor.from_pretrained(
                self.synthetic_config.qwen_model,
                trust_remote_code=True
            )
            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.synthetic_config.qwen_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print(f"  ✓ Qwen2-VL loaded (Free: {self._get_free_memory():.1f}GB)")
        except Exception as e:
            print(f"  ✗ Qwen2-VL failed: {e}")
            self.qwen_model = None
            self.qwen_processor = None
    
    def _load_grounding_dino(self):
        """Load Grounding DINO model."""
        if self.gdino_model is not None:
            return
        
        self._unload_all_models()
        print(f"  Loading Grounding DINO... (Free: {self._get_free_memory():.1f}GB)")
        
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            self.gdino_processor = AutoProcessor.from_pretrained(
                self.synthetic_config.gdino_model
            )
            self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.synthetic_config.gdino_model,
                torch_dtype=torch.float32  # DINO works better with float32
            ).to(self.device).eval()
            
            print(f"  ✓ Grounding DINO loaded (Free: {self._get_free_memory():.1f}GB)")
        except Exception as e:
            print(f"  ✗ Grounding DINO failed: {e}")
            self.gdino_model = None
            self.gdino_processor = None
    
    def _load_sam(self):
        """Load SAM model."""
        if self.sam_model is not None:
            return
        
        self._unload_all_models()
        print(f"  Loading SAM... (Free: {self._get_free_memory():.1f}GB)")
        
        try:
            from transformers import SamModel, SamProcessor
            
            self.sam_processor = SamProcessor.from_pretrained(
                self.synthetic_config.sam_model
            )
            self.sam_model = SamModel.from_pretrained(
                self.synthetic_config.sam_model,
                torch_dtype=torch.float32  # SAM works better with float32
            ).to(self.device).eval()
            
            print(f"  ✓ SAM loaded (Free: {self._get_free_memory():.1f}GB)")
        except Exception as e:
            print(f"  ✗ SAM failed: {e}")
            self.sam_model = None
            self.sam_processor = None
    
    def _load_stable_diffusion(self):
        """Load Stable Diffusion with fallback chain."""
        if self.sd_pipeline is not None:
            return
        
        self._unload_all_models()
        
        # Try SD 3.5 first
        try:
            from diffusers import StableDiffusion3Pipeline
            print(f"  Loading SD 3.5... (Free: {self._get_free_memory():.1f}GB)")
            
            self.sd_pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.synthetic_config.sd_model,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(self.device)
            self.sd_pipeline.enable_attention_slicing()
            
            print(f"  ✓ SD 3.5 loaded (Free: {self._get_free_memory():.1f}GB)")
            return
        except Exception as e:
            print(f"  ⚠ SD 3.5 unavailable: {e}")
        
        # Try SDXL
        try:
            from diffusers import StableDiffusionXLPipeline
            print(f"  Loading SDXL fallback... (Free: {self._get_free_memory():.1f}GB)")
            
            self.sd_pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to(self.device)
            self.sd_pipeline.enable_attention_slicing()
            
            print(f"  ✓ SDXL loaded (Free: {self._get_free_memory():.1f}GB)")
            return
        except Exception as e:
            print(f"  ⚠ SDXL unavailable: {e}")
        
        # Try SD 2.1
        try:
            from diffusers import StableDiffusionPipeline
            print(f"  Loading SD 2.1 fallback... (Free: {self._get_free_memory():.1f}GB)")
            
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16
            ).to(self.device)
            self.sd_pipeline.enable_attention_slicing()
            
            print(f"  ✓ SD 2.1 loaded (Free: {self._get_free_memory():.1f}GB)")
            return
        except Exception as e:
            print(f"  ✗ All SD models failed: {e}")
            self.sd_pipeline = None
    
    def _load_clip(self):
        """Load CLIP for quality scoring."""
        if self.clip_model is not None:
            return
        
        self._unload_all_models()
        print(f"  Loading CLIP... (Free: {self._get_free_memory():.1f}GB)")
        
        try:
            import open_clip
            
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            self.clip_model = model.to(self.device).eval()
            self.clip_preprocess = preprocess
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            print(f"  ✓ CLIP loaded (Free: {self._get_free_memory():.1f}GB)")
        except Exception as e:
            print(f"  ✗ CLIP failed: {e}")
            self.clip_model = None
    
    # =========================================================================
    # Stage 2: Qwen2-VL Scene Analysis
    # =========================================================================
    
    def stage2_scene_analysis(self, image: Image.Image) -> ExtractionResult:
        """Analyze image with Qwen2-VL to extract scene information."""
        start = time.time()
        
        self._load_qwen()
        if self.qwen_model is None:
            return self._fallback_extraction()
        
        try:
            prompt = """Analyze this aerial/satellite image. Provide:
1. SCENE_TYPE: (urban/rural/industrial/residential/agricultural/water/forest/etc.)
2. OBJECT_INVENTORY: object1: count1, object2: count2
3. LAYOUT_DESCRIPTION: spatial arrangement of objects
4. APPEARANCE_DETAILS: colors, textures, materials
5. LIGHTING_CONDITIONS: time of day, shadows"""

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Process with Qwen
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
            )
            
            # Move to device with correct dtype
            model_device = next(self.qwen_model.parameters()).device
            model_dtype = next(self.qwen_model.parameters()).dtype
            
            for k, v in inputs.items():
                if hasattr(v, 'to'):
                    if v.dtype in (torch.float32, torch.float64, torch.bfloat16):
                        inputs[k] = v.to(device=model_device, dtype=model_dtype)
                    else:
                        inputs[k] = v.to(device=model_device)
            
            with torch.no_grad():
                output_ids = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=self.synthetic_config.qwen_max_tokens
                )
            
            response = self.qwen_processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]
            
            result = self._parse_extraction(response)
            
        except Exception as e:
            print(f"    ⚠ Scene analysis error: {e}")
            result = self._fallback_extraction()
        
        self.stage_times["stage_2"].append(time.time() - start)
        return result
    
    def _parse_extraction(self, response: str) -> ExtractionResult:
        """Parse Qwen response into structured result."""
        scene_type = "aerial scene"
        objects: Dict[str, int] = {}
        layout = "Objects distributed across the scene"
        appearance = "Natural colors and textures"
        lighting = "Daylight"
        
        for line in response.split('\n'):
            lower = line.lower().strip()
            if 'scene_type' in lower or 'scene type' in lower:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    scene_type = parts[1].strip().lower()
            elif 'object_inventory' in lower or 'object inventory' in lower:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    objects = self._parse_objects(parts[1])
            elif 'layout' in lower:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    layout = parts[1].strip()
            elif 'appearance' in lower:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    appearance = parts[1].strip()
            elif 'lighting' in lower:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    lighting = parts[1].strip()
        
        return ExtractionResult(
            scene_type=scene_type or "aerial scene",
            object_inventory=objects or {"structure": 1},
            layout_description=layout,
            appearance_details=appearance,
            lighting_conditions=lighting,
            raw_analysis=response
        )
    
    def _parse_objects(self, text: str) -> Dict[str, int]:
        """Parse object inventory string."""
        result = {}
        for part in text.split(','):
            if ':' in part:
                name, count = part.split(':', 1)
                name = name.strip().lower()
                try:
                    result[name] = int(count.strip().split()[0])
                except:
                    result[name] = 1
        return result if result else {"structure": 1}
    
    def _fallback_extraction(self) -> ExtractionResult:
        """Fallback when Qwen is unavailable."""
        return ExtractionResult(
            scene_type="aerial scene",
            object_inventory={"structure": 1},
            layout_description="Objects in aerial view",
            appearance_details="Natural colors",
            lighting_conditions="Daylight",
            raw_analysis="Fallback"
        )
    
    # =========================================================================
    # Stage 3: Grounding DINO Detection
    # =========================================================================
    
    def stage3_detection(self, image: Image.Image, objects: Dict[str, int]) -> DetectionResult:
        """Detect objects with Grounding DINO."""
        start = time.time()
        
        self._load_grounding_dino()
        if self.gdino_model is None:
            return self._fallback_detection()
        
        try:
            # Build text prompt
            text_prompt = ". ".join(objects.keys()) + "."
            
            # Process inputs
            inputs = self.gdino_processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Run model
            with torch.no_grad():
                outputs = self.gdino_model(**inputs)
            
            # Post-process - get raw results first
            target_sizes = torch.tensor([image.size[::-1]], device=self.device)
            results = self.gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=target_sizes
            )[0]
            
            # Filter by threshold manually
            threshold = self.synthetic_config.gdino_box_threshold
            
            boxes = []
            labels = []
            scores_list = []
            
            if 'scores' in results and len(results['scores']) > 0:
                scores = results['scores']
                
                for i, score in enumerate(scores):
                    if score.item() >= threshold:
                        boxes.append(results['boxes'][i].cpu().numpy().tolist())
                        scores_list.append(score.item())
                        # Handle labels
                        if 'labels' in results:
                            if isinstance(results['labels'], list):
                                labels.append(str(results['labels'][i]))
                            else:
                                labels.append("object")
                        else:
                            labels.append("object")
            
            # Compute spatial relationships
            spatial = self._compute_spatial(boxes, labels)
            
            result = DetectionResult(
                boxes=boxes,
                labels=labels,
                scores=scores_list,
                spatial_relationships=spatial
            )
            
        except Exception as e:
            print(f"    ⚠ Detection error: {e}")
            import traceback
            traceback.print_exc()
            result = self._fallback_detection()
        
        self.stage_times["stage_3"].append(time.time() - start)
        return result
    
    def _compute_spatial(self, boxes: List, labels: List) -> List[str]:
        """Compute spatial relationships between objects."""
        relations = []
        for i, (b1, l1) in enumerate(zip(boxes, labels)):
            for j, (b2, l2) in enumerate(zip(boxes, labels)):
                if i >= j:
                    continue
                cx1 = (b1[0] + b1[2]) / 2
                cy1 = (b1[1] + b1[3]) / 2
                cx2 = (b2[0] + b2[2]) / 2
                cy2 = (b2[1] + b2[3]) / 2
                
                if abs(cx1 - cx2) > 50 or abs(cy1 - cy2) > 50:
                    if cx1 < cx2 - 50:
                        relations.append(f"{l1} left of {l2}")
                    elif cx1 > cx2 + 50:
                        relations.append(f"{l1} right of {l2}")
                    elif cy1 < cy2 - 50:
                        relations.append(f"{l1} above {l2}")
                    else:
                        relations.append(f"{l1} below {l2}")
        return relations[:10]
    
    def _fallback_detection(self) -> DetectionResult:
        """Fallback when DINO is unavailable."""
        return DetectionResult(
            boxes=[[100, 100, 400, 400]],
            labels=["structure"],
            scores=[0.5],
            spatial_relationships=[]
        )
    
    # =========================================================================
    # Stage 4: SAM Segmentation
    # =========================================================================
    
    def stage4_segmentation(self, image: Image.Image, detection: DetectionResult) -> SegmentationResult:
        """Segment objects with SAM."""
        start = time.time()
        
        self._load_sam()
        if self.sam_model is None or not detection.boxes:
            return self._fallback_segmentation()
        
        try:
            # Prepare boxes (limit to 10)
            input_boxes = detection.boxes[:10]
            
            # Process inputs
            inputs = self.sam_processor(
                image,
                input_boxes=[input_boxes],
                return_tensors="pt"
            )
            
            # Move to device
            for k, v in inputs.items():
                if hasattr(v, 'to'):
                    inputs[k] = v.to(self.device)
            
            # Run model
            with torch.no_grad():
                outputs = self.sam_model(**inputs)
            
            # Post-process masks
            masks = self.sam_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
                binarize=False
            )
            
            # Extract mask data
            mask_list = []
            areas = []
            
            if masks and len(masks) > 0:
                mask_tensor = masks[0]
                if mask_tensor.dim() >= 2:
                    num_masks = mask_tensor.shape[0] if mask_tensor.dim() > 2 else 1
                    for i in range(min(num_masks, 10)):
                        if mask_tensor.dim() > 3:
                            m = mask_tensor[i, 0].numpy()
                        elif mask_tensor.dim() > 2:
                            m = mask_tensor[i].numpy()
                        else:
                            m = mask_tensor.numpy()
                        m = m.squeeze()
                        mask_list.append(m)
                        areas.append(int((m > 0.5).sum()))
            
            result = SegmentationResult(
                masks=mask_list,
                boundaries=[[] for _ in mask_list],
                areas=areas
            )
            
        except Exception as e:
            print(f"    ⚠ Segmentation error: {e}")
            result = self._fallback_segmentation()
        
        self.stage_times["stage_4"].append(time.time() - start)
        return result
    
    def _fallback_segmentation(self) -> SegmentationResult:
        """Fallback when SAM is unavailable."""
        return SegmentationResult(masks=[], boundaries=[], areas=[10000])
    
    # =========================================================================
    # Stage 5: Prompt Generation
    # =========================================================================
    
    def stage5_prompt(self, extraction: ExtractionResult, detection: DetectionResult, 
                      segmentation: SegmentationResult) -> GeneratedPrompt:
        """Generate SD prompt from extracted information."""
        start = time.time()
        
        # Build object description
        obj_parts = []
        for obj, count in sorted(extraction.object_inventory.items(), 
                                  key=lambda x: x[1], reverse=True):
            obj_parts.append(f"{count} {obj}{'s' if count > 1 else ''}")
        obj_text = ", ".join(obj_parts) if obj_parts else "structures"
        
        # Build spatial description
        spatial = ""
        if detection.spatial_relationships:
            spatial = " " + ". ".join(detection.spatial_relationships[:5]) + "."
        
        # Components
        scene = f"aerial satellite view of {extraction.scene_type} area"
        layout = extraction.layout_description + spatial
        objects = f"containing {obj_text}"
        style = f"{extraction.appearance_details}, {extraction.lighting_conditions}"
        
        # Full prompt
        full = (
            f"High-resolution {scene} {objects}. "
            f"{layout} "
            f"Photorealistic remote sensing imagery, sharp detail, natural colors, "
            f"top-down orthographic view, professional satellite photography."
        )
        
        result = GeneratedPrompt(
            full_prompt=full,
            scene_component=scene,
            layout_component=layout,
            object_component=objects,
            style_component=style
        )
        
        self.stage_times["stage_5"].append(time.time() - start)
        return result
    
    # =========================================================================
    # Stage 6: Stable Diffusion Generation
    # =========================================================================
    
    def stage6_generate(self, prompt: GeneratedPrompt) -> Tuple[Optional[Image.Image], Dict]:
        """Generate image with Stable Diffusion."""
        start = time.time()
        
        self._load_stable_diffusion()
        if self.sd_pipeline is None:
            return None, {"error": "SD not available"}
        
        try:
            params = {
                "prompt": prompt.full_prompt,
                "negative_prompt": "blurry, low quality, distorted, artifacts, watermark, text",
                "num_inference_steps": self.synthetic_config.sd_steps,
                "guidance_scale": self.synthetic_config.sd_guidance_scale,
                "height": self.synthetic_config.sd_image_size,
                "width": self.synthetic_config.sd_image_size,
            }
            
            with torch.no_grad():
                output = self.sd_pipeline(**params)
            
            # Extract image from output
            if hasattr(output, 'images'):
                image = output.images[0]
            elif isinstance(output, tuple):
                image = output[0][0] if isinstance(output[0], list) else output[0]
            else:
                image = output
            
            params["success"] = True
            
        except Exception as e:
            print(f"    ⚠ Generation error: {e}")
            image = None
            params = {"error": str(e), "success": False}
        
        self.stage_times["stage_6"].append(time.time() - start)
        return image, params
    
    # =========================================================================
    # Stages 7-9: Verification (reuse stages 2-4)
    # =========================================================================
    
    def stage7_verify_scene(self, image: Image.Image) -> ExtractionResult:
        """Verify generated image with second-pass analysis."""
        start = time.time()
        result = self.stage2_scene_analysis(image)
        self.stage_times["stage_7"].append(time.time() - start)
        return result
    
    def stage8_verify_detection(self, image: Image.Image, objects: Dict[str, int]) -> DetectionResult:
        """Verify with second-pass detection."""
        start = time.time()
        result = self.stage3_detection(image, objects)
        self.stage_times["stage_8"].append(time.time() - start)
        return result
    
    def stage9_verify_segmentation(self, image: Image.Image, detection: DetectionResult) -> SegmentationResult:
        """Verify with second-pass segmentation."""
        start = time.time()
        result = self.stage4_segmentation(image, detection)
        self.stage_times["stage_9"].append(time.time() - start)
        return result
    
    # =========================================================================
    # Stage 10: Quality Scoring & Refinement
    # =========================================================================
    
    def stage10_score_and_refine(
        self,
        image: Image.Image,
        prompt: GeneratedPrompt,
        orig_ext: ExtractionResult,
        orig_det: DetectionResult,
        verify_ext: ExtractionResult,
        verify_det: DetectionResult
    ) -> Tuple[str, QualityScores]:
        """Compute quality scores and generate refined caption."""
        start = time.time()
        
        self._load_clip()
        
        # CLIP score
        clip_score = self._clip_score(image, prompt.full_prompt)
        
        # Layout IoU
        layout_iou = self._layout_iou(orig_det.boxes, verify_det.boxes)
        
        # Object count accuracy
        obj_acc = self._object_accuracy(orig_ext.object_inventory, verify_ext.object_inventory)
        
        # Check thresholds
        passed = (
            clip_score >= self.synthetic_config.min_clip_score and
            layout_iou >= self.synthetic_config.min_layout_iou and
            obj_acc >= self.synthetic_config.min_object_count_accuracy
        )
        
        scores = QualityScores(
            clip_score=clip_score,
            layout_iou=layout_iou,
            object_count_accuracy=obj_acc,
            all_checks_passed=passed,
            details={"thresholds": {
                "clip": self.synthetic_config.min_clip_score,
                "iou": self.synthetic_config.min_layout_iou,
                "obj": self.synthetic_config.min_object_count_accuracy
            }}
        )
        
        # Generate caption
        caption = self._generate_caption(verify_ext, verify_det)
        
        self.stage_times["stage_10"].append(time.time() - start)
        return caption, scores
    
    def _clip_score(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity."""
        if self.clip_model is None:
            return 0.5
        
        try:
            import torch.nn.functional as F
            
            img_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            txt_input = self.clip_tokenizer([text]).to(self.device)
            
            with torch.no_grad():
                img_feat = self.clip_model.encode_image(img_input)
                txt_feat = self.clip_model.encode_text(txt_input)
                img_feat = F.normalize(img_feat, dim=-1)
                txt_feat = F.normalize(txt_feat, dim=-1)
                sim = (img_feat @ txt_feat.T).item()
            
            return (sim + 1) / 2  # Convert to [0, 1]
        except:
            return 0.5
    
    def _layout_iou(self, boxes1: List, boxes2: List) -> float:
        """Compute average IoU between box sets."""
        if not boxes1 or not boxes2:
            return 0.5
        
        def iou(b1, b2):
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            union = a1 + a2 - inter
            return inter / union if union > 0 else 0
        
        total = 0
        used = set()
        for b1 in boxes1:
            best = 0
            best_idx = -1
            for idx, b2 in enumerate(boxes2):
                if idx not in used:
                    score = iou(b1, b2)
                    if score > best:
                        best = score
                        best_idx = idx
            if best_idx >= 0:
                used.add(best_idx)
                total += best
        
        return total / len(boxes1)
    
    def _object_accuracy(self, orig: Dict[str, int], synth: Dict[str, int]) -> float:
        """Compute object count accuracy."""
        if not orig:
            return 1.0
        total_orig = sum(orig.values())
        total_synth = sum(synth.values())
        if total_orig == 0:
            return 1.0
        return min(total_orig, total_synth) / max(total_orig, total_synth)
    
    def _generate_caption(self, ext: ExtractionResult, det: DetectionResult) -> str:
        """Generate refined caption."""
        obj_parts = []
        for obj, count in sorted(ext.object_inventory.items(), key=lambda x: x[1], reverse=True):
            obj_parts.append(f"{count} {obj}{'s' if count > 1 else ''}")
        obj_text = ", ".join(obj_parts) if obj_parts else "structures"
        
        spatial = ""
        if det.spatial_relationships:
            spatial = " " + ". ".join(det.spatial_relationships[:3]) + "."
        
        return (
            f"Aerial satellite image of {ext.scene_type} area with {obj_text}. "
            f"{ext.layout_description}.{spatial} "
            f"{ext.appearance_details} under {ext.lighting_conditions}."
        )
    
    # =========================================================================
    # Main Processing
    # =========================================================================
    
    def setup(self):
        """Initialize pipeline."""
        print("\n" + "="*70)
        print("SYNTHETIC PIPELINE INITIALIZED")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Free GPU Memory: {self._get_free_memory():.1f} GB")
        print("Models will load on-demand")
        print("="*70)
    
    def load_models(self):
        """Compatibility method - models load lazily."""
        pass
    
    def process_image(self, image_path: Path) -> Optional[SyntheticSample]:
        """Process single image through all stages."""
        try:
            sample_id = str(uuid.uuid4())[:8]
            source = Image.open(image_path).convert('RGB')
            
            # Stage 2: Scene analysis
            print("    Stage 2: Scene analysis...")
            extraction = self.stage2_scene_analysis(source)
            
            # Stage 3: Detection
            print("    Stage 3: Detection...")
            detection = self.stage3_detection(source, extraction.object_inventory)
            
            # Stage 4: Segmentation
            print("    Stage 4: Segmentation...")
            segmentation = self.stage4_segmentation(source, detection)
            
            # Stage 5: Prompt generation
            print("    Stage 5: Prompt generation...")
            prompt = self.stage5_prompt(extraction, detection, segmentation)
            
            # Stage 6: Image generation
            print("    Stage 6: Image generation...")
            synthetic, gen_params = self.stage6_generate(prompt)
            
            if synthetic is None:
                self.failed_count += 1
                return None
            
            # Stage 7-9: Verification
            print("    Stage 7-9: Verification...")
            verify_ext = self.stage7_verify_scene(synthetic)
            verify_det = self.stage8_verify_detection(synthetic, extraction.object_inventory)
            verify_seg = self.stage9_verify_segmentation(synthetic, verify_det)
            
            # Stage 10: Scoring
            print("    Stage 10: Quality scoring...")
            caption, scores = self.stage10_score_and_refine(
                synthetic, prompt, extraction, detection, verify_ext, verify_det
            )
            
            # Save synthetic image
            synth_path = self.synthetic_dir / f"synthetic_{sample_id}.png"
            synthetic.save(synth_path)
            
            # Create sample
            sample = SyntheticSample(
                sample_id=sample_id,
                source_image_path=str(image_path),
                synthetic_image_path=str(synth_path),
                original_extraction=extraction,
                original_detection=detection,
                original_segmentation=segmentation,
                generated_prompt=prompt,
                generation_params=gen_params,
                verification_extraction=verify_ext,
                verification_detection=verify_det,
                verification_segmentation=verify_seg,
                quality_scores=scores,
                refined_caption=caption
            )
            
            # Save metadata
            self._save_metadata(sample)
            
            self.processed_count += 1
            if scores.all_checks_passed:
                self.passed_count += 1
            else:
                self.failed_count += 1
            
            return sample
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            self.failed_count += 1
            return None
    
    # Alias for compatibility
    def process_single_image(self, image_path: Path) -> Optional[SyntheticSample]:
        """Alias for process_image for backward compatibility."""
        return self.process_image(image_path)
    
    def _save_metadata(self, sample: SyntheticSample):
        """Save sample metadata."""
        path = self.metadata_dir / f"{sample.sample_id}.json"
        data = {
            "sample_id": sample.sample_id,
            "source_image_path": sample.source_image_path,
            "synthetic_image_path": sample.synthetic_image_path,
            "original_extraction": asdict(sample.original_extraction),
            "original_detection": asdict(sample.original_detection),
            "generated_prompt": asdict(sample.generated_prompt),
            "generation_params": sample.generation_params,
            "verification_extraction": asdict(sample.verification_extraction),
            "quality_scores": asdict(sample.quality_scores),
            "refined_caption": sample.refined_caption,
            "created_at": sample.created_at,
            "pipeline_version": sample.pipeline_version
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def run(self, source_images: List[Path]) -> Dict[str, Any]:
        """Run pipeline on list of images."""
        print("\n" + "="*70)
        print("SYNTHETIC GENERATION PIPELINE")
        print("="*70)
        print(f"Images: {len(source_images)}")
        print(f"Target: {self.synthetic_config.target_synthetic_count}")
        print(f"Thresholds: CLIP>{self.synthetic_config.min_clip_score}, "
              f"IoU>{self.synthetic_config.min_layout_iou}, "
              f"Obj>{self.synthetic_config.min_object_count_accuracy}")
        print("="*70)
        
        samples = []
        for i, path in enumerate(source_images):
            if self.passed_count >= self.synthetic_config.target_synthetic_count:
                print(f"\n✓ Target reached: {self.passed_count}")
                break
            
            print(f"\n[{i+1}/{len(source_images)}] {path.name}")
            sample = self.process_image(path)
            
            if sample and sample.quality_scores.all_checks_passed:
                samples.append(sample)
                print(f"  ✓ Passed - CLIP:{sample.quality_scores.clip_score:.3f} "
                      f"IoU:{sample.quality_scores.layout_iou:.3f}")
            else:
                print(f"  ✗ Failed quality check")
            
            if (i + 1) % 10 == 0:
                rate = self.passed_count / max(1, self.processed_count) * 100
                print(f"\n--- Progress: {self.passed_count}/{self.processed_count} ({rate:.1f}%) ---")
        
        # Summary
        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        print(f"Processed: {self.processed_count}")
        print(f"Passed: {self.passed_count}")
        print(f"Failed: {self.failed_count}")
        print(f"Rate: {self.passed_count/max(1,self.processed_count)*100:.1f}%")
        
        # Timing
        print("\nStage Timing (avg):")
        for stage, times in self.stage_times.items():
            if times:
                print(f"  {stage}: {sum(times)/len(times):.2f}s")
        
        return {
            "processed": self.processed_count,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "quality_rate": self.passed_count / max(1, self.processed_count),
            "samples": samples
        }
    
    def generate_dataset(self, source_dir: Path, output_dir: Path, 
                         target_count: Optional[int] = None) -> Dict[str, Any]:
        """Generate synthetic dataset from source directory."""
        if target_count:
            self.synthetic_config.target_synthetic_count = target_count
        
        # Collect images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            images.extend(source_dir.glob(ext))
            images.extend(source_dir.glob(ext.upper()))
        
        if not images:
            print(f"No images in {source_dir}")
            return {"generated": 0}
        
        # Run
        result = self.run(images)
        
        # Save manifest
        manifest_path = self.metadata_dir / "dataset.json"
        manifest = {
            "pairs": [{
                "image_path": str(s.synthetic_image_path),
                "caption": s.refined_caption,
                "quality_scores": asdict(s.quality_scores),
                "image_id": s.sample_id
            } for s in result.get("samples", [])]
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return {
            "generated": result.get("passed", 0),
            "quality_rate": result.get("quality_rate", 0.0),
            "manifest_path": str(manifest_path)
        }
