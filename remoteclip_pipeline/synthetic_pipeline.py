#!/usr/bin/env python3
# type: ignore
"""
Synthetic Generation Pipeline for Remote Sensing Images (A100 x2 build)

6-stage pipeline, tuned for 2x NVIDIA A100 (80GB) + 7 CPU cores:
1. Qwen2.5-VL-7B dense scene analysis        (analysis_device, e.g. cuda:1)
2. Grounding DINO layout/object detection    (analysis_device)
3. Prompt-variant generation (N prompts)     (CPU)
4. FLUX.1-dev / SD3.5 image generation       (gen_device, e.g. cuda:0)
5. Qwen2.5-VL multi-caption generation       (analysis_device)
6. Manifest & metadata assembly              (CPU)

Fan-out per source image: N_PROMPTS prompt variants -> N_PROMPTS synthetic
images -> each image gets N_CAPTIONS distinct captions. With the default
config (10,000 source images x 5 prompts x 5 captions) this produces
50,000 synthetic images / 250,000 image-caption pairs.

Analysis (stage 1-3, GPU1) for the *next* source image is prefetched on a
background thread while generation + captioning (GPU0/GPU1) for the
*current* source image runs, so both GPUs stay busy concurrently.

No CPU-offload, no 8-bit quantization: both A100s have ample VRAM for
these models running natively in bf16.
"""
from __future__ import annotations

import gc
import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# CPU thread defaults (overridden at runtime from SyntheticConfig.cpu_threads).
# 7 cores available; leave 1 free for OS/IO by default.
_DEFAULT_CPU_THREADS = "6"
os.environ.setdefault('OMP_NUM_THREADS', _DEFAULT_CPU_THREADS)
os.environ.setdefault('MKL_NUM_THREADS', _DEFAULT_CPU_THREADS)
os.environ.setdefault('OPENBLAS_NUM_THREADS', _DEFAULT_CPU_THREADS)
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', _DEFAULT_CPU_THREADS)
os.environ.setdefault('NUMEXPR_NUM_THREADS', _DEFAULT_CPU_THREADS)

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
class GeneratedPrompt:
    """A single generated prompt variant for the image generator."""
    full_prompt: str
    scene_component: str
    layout_component: str
    object_component: str
    style_component: str
    variant_index: int = 0


@dataclass
class GeneratedImageRecord:
    """One generated image plus its N captions."""
    image_id: str
    image_path: str
    prompt: GeneratedPrompt
    generation_params: Dict[str, Any]
    captions: List[str] = field(default_factory=list)


@dataclass
class SyntheticSample:
    """Complete synthetic sample: one source image -> N generated images."""
    sample_id: str
    source_image_path: str
    original_extraction: ExtractionResult
    original_detection: DetectionResult
    generated_images: List[GeneratedImageRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = "3.0.0"


# =============================================================================
# Main Pipeline Class
# =============================================================================

class SyntheticGenerationPipeline:
    """
    Synthetic Generation Pipeline with upfront model loading, split across
    two GPUs: the image generator on `gen_device`, and the VLM captioner +
    detector on `analysis_device`. All models loaded natively in bf16 with
    no CPU offload, since both A100s have 80GB of VRAM to spare.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.synthetic_config = config.synthetic
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Resolve multi-GPU placement, degrading gracefully to a single GPU
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_gpus >= 2:
            self.gen_device = torch.device(self.synthetic_config.gen_device)
            self.analysis_device = torch.device(self.synthetic_config.analysis_device)
        elif n_gpus == 1:
            self.gen_device = self.device
            self.analysis_device = self.device
        else:
            self.gen_device = torch.device("cpu")
            self.analysis_device = torch.device("cpu")

        # Apply CPU thread budget from config (default 6 of 7 cores)
        torch.set_num_threads(self.synthetic_config.cpu_threads)
        torch.set_num_interop_threads(max(1, self.synthetic_config.cpu_threads // 2))

        # Model references (loaded during initialization)
        self.qwen_model = None
        self.qwen_processor = None
        self.gdino_model = None
        self.gdino_processor = None
        self.sd_pipeline = None

        # Statistics
        self.processed_count = 0  # source images processed
        self.passed_count = 0     # synthetic images generated
        self.failed_count = 0
        self.stage_times: Dict[str, List[float]] = {
            f"stage_{i}": [] for i in range(1, 7)
        }

        # Output paths
        self.output_root = config.output_dir.parent
        self.synthetic_dir = self.output_root / "images"
        self.metadata_dir = self.output_root / "metadata"
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Single background worker for analysis prefetch (keeps GPU1 busy
        # while GPU0 generates images for the previous source image)
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1)

        # Load all models upfront
        self._load_all_models()

    # =========================================================================
    # GPU Memory Management
    # =========================================================================

    def _clear_gpu_memory(self):
        """Clear GPU memory (lightweight; models stay resident on both GPUs)."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_free_memory(self, device: Optional[torch.device] = None) -> float:
        """Get free GPU memory in GB for a given device (default: gen_device)."""
        if not torch.cuda.is_available():
            return 0.0
        dev = device if device is not None else self.gen_device
        idx = dev.index if dev.index is not None else 0
        free, total = torch.cuda.mem_get_info(idx)
        return free / (1024**3)

    # =========================================================================
    # Model Loading (Upfront, split across both GPUs)
    # =========================================================================

    def _load_all_models(self):
        """Load all models at initialization, placed across both GPUs."""
        print("\nLoading models (this may take a few minutes)...")
        print(f"  Generator device:  {self.gen_device}")
        print(f"  Analysis device:   {self.analysis_device}")

        self._load_qwen()
        self._load_grounding_dino()
        self._load_stable_diffusion()

        print(f"\n{'='*70}")
        print("SYNTHETIC PIPELINE INITIALIZED")
        print(f"{'='*70}")
        print(f"Generator device: {self.gen_device} "
              f"(Free: {self._get_free_memory(self.gen_device):.1f} GB)")
        print(f"Analysis device:  {self.analysis_device} "
              f"(Free: {self._get_free_memory(self.analysis_device):.1f} GB)")
        print("All models loaded natively in bf16, no CPU offload")
        print(f"{'='*70}\n")

    def _load_qwen(self):
        """Load Qwen2.5-VL-7B (bf16, native, on analysis_device)."""
        dtype = getattr(torch, self.synthetic_config.qwen_dtype)
        print(f"  Loading {self.synthetic_config.qwen_model} "
              f"({self.synthetic_config.qwen_dtype})... "
              f"(Free: {self._get_free_memory(self.analysis_device):.1f}GB)")

        self._clear_gpu_memory()

        try:
            from transformers import AutoProcessor
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
            except ImportError:
                # Fallback for older transformers without Qwen2.5-VL support
                from transformers import Qwen2VLForConditionalGeneration as QwenVLModel

            self.qwen_processor = AutoProcessor.from_pretrained(
                self.synthetic_config.qwen_model,
                trust_remote_code=True
            )

            self.qwen_model = QwenVLModel.from_pretrained(
                self.synthetic_config.qwen_model,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.analysis_device).eval()

            self._clear_gpu_memory()

            print(f"  \u2713 Qwen loaded on {self.analysis_device} "
                  f"(Free: {self._get_free_memory(self.analysis_device):.1f}GB)")
        except Exception as e:
            print(f"  \u2717 Qwen load failed: {e}")
            raise

    def _load_grounding_dino(self):
        """Load Grounding DINO (bf16, native, on analysis_device)."""
        dtype = getattr(torch, self.synthetic_config.gdino_dtype)
        print(f"  Loading Grounding DINO ({self.synthetic_config.gdino_dtype})... "
              f"(Free: {self._get_free_memory(self.analysis_device):.1f}GB)")

        self._clear_gpu_memory()

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self.gdino_processor = AutoProcessor.from_pretrained(
                self.synthetic_config.gdino_model
            )
            self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.synthetic_config.gdino_model,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            ).to(self.analysis_device).eval()

            self._clear_gpu_memory()

            print(f"  \u2713 Grounding DINO loaded on {self.analysis_device} "
                  f"(Free: {self._get_free_memory(self.analysis_device):.1f}GB)")
        except Exception as e:
            print(f"  \u2717 Grounding DINO load failed: {e}")
            raise

    def _load_stable_diffusion(self):
        """Load the image generator (FLUX.1-dev or SD3.5) natively on gen_device.

        No `enable_sequential_cpu_offload()` / `enable_model_cpu_offload()`:
        those exist for <16GB GPUs and would serialize every forward pass
        through host RAM. An A100-80GB fits these models whole, so we place
        the pipeline on the GPU once and keep it resident.
        """
        self._clear_gpu_memory()
        dtype = getattr(torch, self.synthetic_config.sd_dtype)

        try:
            backend = self.synthetic_config.sd_backend
            if backend == "flux":
                from diffusers import FluxPipeline
                print(f"  Loading {self.synthetic_config.sd_model} "
                      f"({self.synthetic_config.sd_dtype})... "
                      f"(Free: {self._get_free_memory(self.gen_device):.1f}GB)")
                self.sd_pipeline = FluxPipeline.from_pretrained(
                    self.synthetic_config.sd_model,
                    torch_dtype=dtype,
                )
            else:
                from diffusers import StableDiffusion3Pipeline
                print(f"  Loading {self.synthetic_config.sd_model} "
                      f"({self.synthetic_config.sd_dtype})... "
                      f"(Free: {self._get_free_memory(self.gen_device):.1f}GB)")
                self.sd_pipeline = StableDiffusion3Pipeline.from_pretrained(
                    self.synthetic_config.sd_model,
                    torch_dtype=dtype,
                )

            # Full native placement on a single A100 - no offloading required
            self.sd_pipeline = self.sd_pipeline.to(self.gen_device)

            self._clear_gpu_memory()

            print(f"  \u2713 Generator loaded on {self.gen_device} "
                  f"(Free: {self._get_free_memory(self.gen_device):.1f}GB)")
        except Exception as e:
            print(f"  \u2717 Generator load failed: {e}")
            self.sd_pipeline = None
            raise

    # =========================================================================
    # Stage 1: Qwen2.5-VL Scene Analysis
    # =========================================================================

    def stage1_scene_analysis(self, image: Image.Image) -> ExtractionResult:
        """Analyze image with Qwen2.5-VL to extract scene information."""
        start = time.time()

        if self.qwen_model is None:
            return self._fallback_extraction()

        try:
            prompt = """Analyze this aerial/satellite image in detail. Provide:
1. SCENE_TYPE: (urban/rural/industrial/residential/agricultural/water/forest/etc.)
2. OBJECT_INVENTORY: List ALL visible objects with accurate counts (e.g., "buildings: 45, roads: 8, vehicles: 12, trees: 20, parking lots: 3"). Be specific - identify buildings, roads, vehicles, trees, water bodies, bridges, parking areas, etc.
3. LAYOUT_DESCRIPTION: Describe the spatial arrangement and how objects are organized (e.g., "buildings arranged in grid pattern, roads forming intersection, vehicles parked along streets")
4. KEY_FEATURES: Notable landmarks or distinctive elements"""

            response = self._qwen_generate(image, prompt, max_new_tokens=self.synthetic_config.qwen_max_tokens)
            result = self._parse_extraction(response)

        except Exception as e:
            print(f"    \u26a0 Scene analysis error: {e}")
            result = self._fallback_extraction()

        self.stage_times["stage_1"].append(time.time() - start)
        return result

    def _qwen_generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Run a single-turn Qwen VLM generation for one image."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]

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

        model_dtype = next(self.qwen_model.parameters()).dtype
        for k, v in inputs.items():
            if hasattr(v, 'to'):
                if v.dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
                    inputs[k] = v.to(device=self.analysis_device, dtype=model_dtype)
                else:
                    inputs[k] = v.to(device=self.analysis_device)

        with torch.no_grad():
            output_ids = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        response = self.qwen_processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]

        del output_ids, inputs
        return response

    def _parse_extraction(self, response: str) -> ExtractionResult:
        """Parse Qwen response into structured result."""
        scene_type = "aerial scene"
        objects: Dict[str, int] = {}
        layout = "Objects distributed across the scene"
        appearance = "Realistic aerial view"
        lighting = "Natural lighting"

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
            elif 'layout' in lower and 'spatial' not in lower:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    layout = parts[1].strip()
            elif 'key_features' in lower or 'key features' in lower:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    appearance = parts[1].strip()

        return ExtractionResult(
            scene_type=scene_type or "aerial scene",
            object_inventory=objects or {"structures": 1},
            layout_description=layout,
            appearance_details=appearance,
            lighting_conditions=lighting,
            raw_analysis=response
        )

    def _parse_objects(self, text: str) -> Dict[str, int]:
        """Parse object inventory string - handles multiple formats."""
        result = {}
        parts = text.replace(';', ',').split(',')
        for part in parts:
            part = part.strip()
            if ':' in part:
                name, count = part.split(':', 1)
                name = name.strip().lower()
                try:
                    result[name] = int(count.strip().split()[0])
                except Exception:
                    result[name] = 1
            elif any(char.isdigit() for char in part):
                words = part.split()
                for i, word in enumerate(words):
                    if word.isdigit():
                        count = int(word)
                        name_words = [w for j, w in enumerate(words) if j != i]
                        name = ' '.join(name_words).strip().lower()
                        if name:
                            result[name] = count
                        break
        return result if result else {"structures": 1}

    def _fallback_extraction(self) -> ExtractionResult:
        """Fallback when Qwen is unavailable."""
        return ExtractionResult(
            scene_type="aerial scene",
            object_inventory={"structures": 1},
            layout_description="Objects in aerial view",
            appearance_details="Realistic aerial view",
            lighting_conditions="Natural lighting",
            raw_analysis="Fallback"
        )

    # =========================================================================
    # Stage 2: Grounding DINO Detection
    # =========================================================================

    def stage2_detection(self, image: Image.Image, objects: Dict[str, int]) -> DetectionResult:
        """Detect objects with Grounding DINO."""
        start = time.time()

        if self.gdino_model is None:
            return self._fallback_detection()

        try:
            text_prompt = ". ".join(objects.keys()) + "."

            inputs = self.gdino_processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self.analysis_device)

            model_dtype = next(self.gdino_model.parameters()).dtype
            if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "to"):
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

            with torch.no_grad():
                outputs = self.gdino_model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]], device=self.analysis_device)
            results = self.gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=target_sizes
            )[0]

            threshold = self.synthetic_config.gdino_box_threshold

            boxes = []
            labels = []
            scores_list = []

            if 'scores' in results and len(results['scores']) > 0:
                scores = results['scores']
                for i, score in enumerate(scores):
                    if score.item() >= threshold:
                        boxes.append(results['boxes'][i].float().cpu().numpy().tolist())
                        scores_list.append(score.item())
                        if 'labels' in results:
                            if isinstance(results['labels'], list):
                                labels.append(str(results['labels'][i]))
                            else:
                                labels.append("object")
                        else:
                            labels.append("object")

            spatial = self._compute_spatial(boxes, labels)

            del outputs, inputs, target_sizes, results

            result = DetectionResult(
                boxes=boxes,
                labels=labels,
                scores=scores_list,
                spatial_relationships=spatial
            )

        except Exception as e:
            print(f"    \u26a0 Detection error: {e}")
            result = self._fallback_detection()

        self.stage_times["stage_2"].append(time.time() - start)
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
    # Stage 3: Prompt Variant Generation (N prompts per source image)
    # =========================================================================

    # Style suffixes used to diversify the N prompt variants while keeping
    # the underlying scene/object/layout content identical.
    _STYLE_VARIANTS = [
        "High resolution satellite imagery, sharp focus, natural colors, top-down orthographic view.",
        "Ultra-detailed aerial photograph, crisp edges, true-to-life colors, nadir viewing angle.",
        "Professional remote sensing imagery, fine detail, balanced exposure, straight-down perspective.",
        "High-fidelity orbital photograph, clear atmosphere, accurate colors, vertical overhead view.",
        "Detailed geospatial imagery, sharp resolution, realistic lighting, direct overhead shot.",
    ]

    def stage3_prompt_variants(
        self, extraction: ExtractionResult, detection: DetectionResult
    ) -> List[GeneratedPrompt]:
        """Generate N diverse prompt variants for the image generator."""
        start = time.time()

        n = self.synthetic_config.n_prompts_per_image
        items = sorted(extraction.object_inventory.items(), key=lambda x: x[1], reverse=True)

        prompts: List[GeneratedPrompt] = []
        for i in range(n):
            # Rotate object emphasis order per variant for content diversity
            rotated = items[i % max(1, len(items)):] + items[:i % max(1, len(items))] if items else items
            obj_parts = [f"{count} {obj}{'s' if count > 1 else ''}" for obj, count in rotated]
            obj_text = ", ".join(obj_parts) if obj_parts else "structures"

            spatial = ""
            if detection.spatial_relationships:
                offset = i % max(1, len(detection.spatial_relationships))
                rotated_spatial = detection.spatial_relationships[offset:] + detection.spatial_relationships[:offset]
                spatial = " " + ". ".join(rotated_spatial[:5]) + "."

            full = f"Realistic aerial satellite photograph of {extraction.scene_type} area"
            if obj_text:
                full += f" with {obj_text}"
            if extraction.layout_description:
                full += f". {extraction.layout_description}"
            if spatial:
                full += spatial
            if extraction.appearance_details and extraction.appearance_details != "Realistic aerial view":
                full += f". Notable features: {extraction.appearance_details}"

            style = self._STYLE_VARIANTS[i % len(self._STYLE_VARIANTS)]
            full += f" {style}"

            prompts.append(GeneratedPrompt(
                full_prompt=full,
                scene_component=f"aerial satellite view of {extraction.scene_type} area",
                layout_component=extraction.layout_description + spatial,
                object_component=f"containing {obj_text}" if obj_text else "structures",
                style_component=style,
                variant_index=i,
            ))

        self.stage_times["stage_3"].append(time.time() - start)
        return prompts

    # =========================================================================
    # Stage 4: Image Generation (N images per source image)
    # =========================================================================

    def stage4_generate(self, prompt: GeneratedPrompt, seed: int) -> Tuple[Optional[Image.Image], Dict]:
        """Generate one image with the FLUX/SD3.5 generator."""
        start = time.time()

        if self.sd_pipeline is None:
            return None, {"error": "Generator not available", "success": False}

        try:
            generator = torch.Generator(device=self.gen_device).manual_seed(seed)
            params = {
                "prompt": prompt.full_prompt,
                "num_inference_steps": self.synthetic_config.sd_steps,
                "guidance_scale": self.synthetic_config.sd_guidance_scale,
                "height": self.synthetic_config.sd_image_size,
                "width": self.synthetic_config.sd_image_size,
                "generator": generator,
            }
            # FLUX (distilled, guidance baked in) does not take a negative
            # prompt the way SD3.5's CFG does.
            if self.synthetic_config.sd_backend != "flux":
                params["negative_prompt"] = "blurry, low quality, distorted, artifacts, watermark, text"

            with torch.no_grad():
                output = self.sd_pipeline(**params)

            if hasattr(output, 'images'):
                image = output.images[0]
            elif isinstance(output, tuple):
                image = output[0][0] if isinstance(output[0], list) else output[0]
            else:
                image = output

            log_params = {k: v for k, v in params.items() if k != "generator"}
            log_params["seed"] = seed
            log_params["success"] = True

        except Exception as e:
            print(f"    \u26a0 Generation error: {e}")
            image = None
            log_params = {"error": str(e), "success": False, "seed": seed}

        self.stage_times["stage_4"].append(time.time() - start)
        return image, log_params

    # =========================================================================
    # Stage 5: Multi-Caption Generation (M captions per generated image)
    # =========================================================================

    def stage5_captions(self, image: Image.Image, extraction: ExtractionResult) -> List[str]:
        """Ask Qwen2.5-VL for N diverse captions describing the actual generated image."""
        start = time.time()

        n = self.synthetic_config.n_captions_per_image
        captions: List[str] = []

        if self.qwen_model is not None:
            try:
                prompt = (
                    f"Write {n} diverse, natural-language captions describing this aerial/satellite "
                    f"image, for use in training an image-text retrieval model. "
                    f"Each caption must be a single, information-dense sentence. "
                    f"Vary emphasis across captions (scene type, dominant objects and counts, "
                    f"spatial layout, visual style). Output exactly {n} lines, each formatted as:\n"
                    f"CAPTION_i: <caption text>"
                )
                response = self._qwen_generate(image, prompt, max_new_tokens=64 * n)
                captions = self._parse_captions(response, n)
            except Exception as e:
                print(f"    \u26a0 Captioning error: {e}")
                captions = []

        # Pad with deterministic template captions if the VLM under-produced
        while len(captions) < n:
            captions.append(self._template_caption(extraction, variant=len(captions)))
        captions = captions[:n]

        self.stage_times["stage_5"].append(time.time() - start)
        return captions

    def _parse_captions(self, response: str, n: int) -> List[str]:
        """Parse 'CAPTION_i: ...' lines out of a Qwen response."""
        captions = []
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            if ':' in line:
                head, tail = line.split(':', 1)
                if 'caption' in head.lower() and tail.strip():
                    captions.append(tail.strip())

        if not captions:
            # Model didn't follow the "CAPTION_i:" format - fall back to
            # treating each non-empty line as its own caption.
            captions = [l.strip('- ').strip() for l in response.split('\n') if l.strip()]

        return captions[:n]

    def _template_caption(self, ext: ExtractionResult, variant: int = 0) -> str:
        """Deterministic fallback caption built from structured extraction."""
        items = sorted(ext.object_inventory.items(), key=lambda x: x[1], reverse=True)
        if items:
            offset = variant % len(items)
            rotated = items[offset:] + items[:offset]
            obj_parts = [f"{count} {obj}{'s' if count > 1 else ''}" for obj, count in rotated[:5]]
            obj_text = ", ".join(obj_parts)
        else:
            obj_text = "structures"
        return (
            f"Aerial satellite image of {ext.scene_type} area with {obj_text}. "
            f"{ext.layout_description}. {ext.appearance_details} under {ext.lighting_conditions}."
        )

    # =========================================================================
    # Stage 6: Metadata persistence
    # =========================================================================

    def _save_metadata(self, sample: SyntheticSample):
        """Save per-source-image metadata (all generated images + captions)."""
        start = time.time()
        path = self.metadata_dir / f"{sample.sample_id}.json"
        data = {
            "sample_id": sample.sample_id,
            "source_image_path": sample.source_image_path,
            "original_extraction": asdict(sample.original_extraction),
            "original_detection": asdict(sample.original_detection),
            "generated_images": [asdict(r) for r in sample.generated_images],
            "created_at": sample.created_at,
            "pipeline_version": sample.pipeline_version
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self.stage_times["stage_6"].append(time.time() - start)

    # =========================================================================
    # Main Processing
    # =========================================================================

    def setup(self):
        """Initialize pipeline (models already loaded in __init__)."""
        print("\n" + "="*70)
        print("SYNTHETIC PIPELINE READY")
        print("="*70)
        print(f"Generator device: {self.gen_device}")
        print(f"Analysis device:  {self.analysis_device}")
        print("="*70)

    def load_models(self):
        """Compatibility no-op - models are loaded eagerly in __init__."""
        pass

    def analyze_source(self, image_path: Path) -> Tuple[Image.Image, ExtractionResult, DetectionResult, List[GeneratedPrompt]]:
        """Stages 1-3: scene analysis, detection, prompt-variant generation.

        Runs entirely on `analysis_device` (GPU1) and is safe to call from
        the background prefetch thread while GPU0 generates images for a
        different source image.
        """
        source = Image.open(image_path).convert('RGB')
        extraction = self.stage1_scene_analysis(source)
        detection = self.stage2_detection(source, extraction.object_inventory)
        prompts = self.stage3_prompt_variants(extraction, detection)
        return source, extraction, detection, prompts

    def process_source_image(
        self,
        image_path: Path,
        precomputed: Optional[Tuple[Image.Image, ExtractionResult, DetectionResult, List[GeneratedPrompt]]] = None,
    ) -> Optional[SyntheticSample]:
        """Process one source image through stages 4-6, producing
        n_prompts_per_image synthetic images each with n_captions_per_image
        captions.
        """
        try:
            sample_id = str(uuid.uuid4())[:8]

            if precomputed is not None:
                _source, extraction, detection, prompts = precomputed
            else:
                _source, extraction, detection, prompts = self.analyze_source(image_path)

            base_seed = random.randint(0, 2**31 - 1)

            image_records: List[GeneratedImageRecord] = []
            for prompt in prompts:
                image, gen_params = self.stage4_generate(prompt, seed=base_seed + prompt.variant_index)

                if image is None:
                    self.failed_count += 1
                    continue

                captions = self.stage5_captions(image, extraction)

                image_id = f"{sample_id}_{prompt.variant_index}"
                synth_path = self.synthetic_dir / f"synthetic_{image_id}.png"
                image.save(synth_path)

                image_records.append(GeneratedImageRecord(
                    image_id=image_id,
                    image_path=str(synth_path),
                    prompt=prompt,
                    generation_params=gen_params,
                    captions=captions,
                ))
                self.passed_count += 1

            self.processed_count += 1

            sample = SyntheticSample(
                sample_id=sample_id,
                source_image_path=str(image_path),
                original_extraction=extraction,
                original_detection=detection,
                generated_images=image_records,
            )

            self._save_metadata(sample)
            self._clear_gpu_memory()

            return sample

        except Exception as e:
            print(f"    \u2717 Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            self.failed_count += 1
            self._clear_gpu_memory()
            return None

    # Alias for backward compatibility
    def process_image(self, image_path: Path) -> Optional[SyntheticSample]:
        return self.process_source_image(image_path)

    def run(self, source_images: List[Path]) -> Dict[str, Any]:
        """Run pipeline over source images, prefetching analysis for image
        i+1 (GPU1) while generating + captioning image i (GPU0/GPU1)."""
        target_images = self.synthetic_config.target_synthetic_count

        print("\n" + "="*70)
        print("SYNTHETIC GENERATION PIPELINE")
        print("="*70)
        print(f"Source images available: {len(source_images)}")
        print(f"Prompts/image: {self.synthetic_config.n_prompts_per_image}  "
              f"Captions/image: {self.synthetic_config.n_captions_per_image}")
        print(f"Target synthetic images: {target_images}  "
              f"(-> {target_images * self.synthetic_config.n_captions_per_image} pairs)")
        print("Mode: LoRA Training (all generated images accepted)")
        print("="*70)

        samples: List[SyntheticSample] = []
        n = len(source_images)

        if n == 0:
            return {"processed": 0, "passed": 0, "failed": 0, "quality_rate": 0.0, "samples": []}

        futures = {0: self._prefetch_executor.submit(self.analyze_source, source_images[0])}

        i = 0
        while i < n:
            if self.passed_count >= target_images:
                print(f"\n\u2713 Target reached: {self.passed_count} images")
                break

            precomputed = futures.pop(i).result()

            if (i + 1) < n:
                futures[i + 1] = self._prefetch_executor.submit(self.analyze_source, source_images[i + 1])

            print(f"\n[{i+1}/{n}] {source_images[i].name}")
            sample = self.process_source_image(source_images[i], precomputed=precomputed)

            if sample:
                samples.append(sample)
                print(f"  \u2713 {len(sample.generated_images)} images generated "
                      f"({self.synthetic_config.n_captions_per_image} captions each)")
            else:
                print("  \u2717 Generation failed")

            if (i + 1) % 10 == 0:
                print(f"\n--- Progress: {self.passed_count}/{target_images} images "
                      f"({self.processed_count} source images processed) ---")

            i += 1

        # Cancel any pending prefetch
        for f in futures.values():
            f.cancel()

        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        print(f"Source images processed: {self.processed_count}")
        print(f"Synthetic images generated: {self.passed_count}")
        print(f"Failed generations: {self.failed_count}")

        print("\nStage Timing (avg):")
        for stage, times in self.stage_times.items():
            if times:
                print(f"  {stage}: {sum(times)/len(times):.2f}s")

        return {
            "processed": self.processed_count,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "quality_rate": 1.0 if self.passed_count else 0.0,
            "samples": samples
        }

    def generate_dataset(self, source_images_dir: Path = None, output_dir: Path = None,
                         target_count: Optional[int] = None, source_dir: Path = None) -> Dict[str, Any]:
        """Generate synthetic dataset from source directory.

        `target_count` (if given) overrides the number of synthetic *images*
        to generate (not pairs). Total pairs = images * n_captions_per_image.
        """
        source_path = source_images_dir or source_dir
        if source_path is None:
            raise ValueError("Must provide source_images_dir or source_dir")

        if target_count:
            self.synthetic_config.target_synthetic_count = target_count

        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            images.extend(source_path.glob(ext))
            images.extend(source_path.glob(ext.upper()))

        if not images:
            print(f"No images in {source_path}")
            return {"generated": 0, "pairs_generated": 0, "samples": []}

        # Respect n_source_images cap if fewer/more are available than requested
        max_sources = self.synthetic_config.n_source_images
        if max_sources and len(images) > max_sources:
            images = images[:max_sources]

        result = self.run(images)

        manifest_path = self.metadata_dir / "dataset.json"
        samples = result.get("samples", [])

        pairs = []
        for s in samples:
            for rec in s.generated_images:
                for cap_idx, caption in enumerate(rec.captions):
                    pairs.append({
                        "image_path": rec.image_path,
                        "caption": caption,
                        "image_id": rec.image_id,
                        "caption_id": f"{rec.image_id}_{cap_idx}",
                        "sample_id": s.sample_id,
                        "quality_scores": {"clip_score": 1.0},
                    })

        manifest = {"pairs": pairs}
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\nManifest: {len(samples)} source images -> "
              f"{result.get('passed', 0)} synthetic images -> {len(pairs)} image-caption pairs")
        print(f"Saved to: {manifest_path}")

        return {
            "generated": result.get("passed", 0),
            "pairs_generated": len(pairs),
            "quality_rate": result.get("quality_rate", 0.0),
            "manifest_path": str(manifest_path),
            "samples": samples
        }
