#!/usr/bin/env python3
"""
COMPLETE PIPELINE: Qwen2-VL → Grounding DINO → SAM → Template Combiner → SDXL+ControlNet
Optimized for GPU server execution

Output: Generated image + detailed description
"""

import os
import sys
import json
import torch
import gc
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# SERVER GPU CHECK
if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU detected! This pipeline requires GPU.")
    sys.exit(1)

device = torch.device('cuda')
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")

# Set GPU memory management
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'

print("=" * 80)
print("ENHANCED PIPELINE: Qwen2-VL → DINO → SAM → SDXL+ControlNet")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Use environment variables for server deployment
    INPUT_DIR = Path(os.getenv("INPUT_DIR", "datasets/rsicd_images"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/pipeline_results"))
    # Use default HuggingFace cache on C: drive to save D: space
    CACHE_DIR = Path(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    
    # Models
    UPSCALER_MODEL = "PyTorch-Bicubic"  # High-quality bicubic upscaling
    SAM_MODEL = "facebook/sam-vit-base"  # Precise localization
    QWEN2VL_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"
    SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_MODEL = "diffusers/controlnet-canny-sdxl-1.0"  # Canny edge control
    
    # Settings
    MAX_IMAGES = 1
    UPSCALE_FACTOR = 4
    OUTPUT_IMAGE_SIZE = 1024
    NUM_INFERENCE_STEPS = 35  # Slightly more steps for better quality
    GUIDANCE_SCALE = 7.5  # Lower = more natural variation, less "forced" look
    CONTROLNET_CONDITIONING_SCALE = 0.6  # Balance structure and realistic textures
    # Increase this to 9.0-10.0 for even more prompt adherence (may reduce creativity)
    MAX_FINAL_PROMPT_WORDS = 300  # Increased for detailed SD3.5 prompts
    
    # Detection classes - expanded for better aerial imagery detection
    DETECT_CLASSES = [
        "building", "house", "road", "street", "highway", "vehicle", "car",
        "tree", "forest", "water", "river", "pond", "lake", "field", "farmland",
        "airplane", "ship", "boat", "bridge", "parking lot", "factory", "warehouse",
        "residential area", "commercial area", "industrial area", "park", "garden",
        "railway", "stadium", "airport", "harbor", "port"
    ]
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / "images").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "metadata").mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STAGE 0: Bicubic Upscaler
# ============================================================================

class RealESRGANUpscaler:
    def __init__(self, config):
        self.config = config
        print("\n[1/6] Loading Bicubic Upscaler...")
        
        try:
            import torch.nn.functional as F
            
            self.upscale_factor = config.UPSCALE_FACTOR
            self.device = device
            
            print(f"✓ Bicubic upscaler loaded (upscale factor: {config.UPSCALE_FACTOR}x)")
            
        except Exception as e:
            print(f"Error loading upscaler: {e}")
            raise
    
    def upscale_image(self, image):
        """Upscale a single image using high-quality bicubic interpolation"""
        try:
            import torchvision.transforms as T
            import torch.nn.functional as F
            
            # Get original size
            orig_w, orig_h = image.size
            new_w = orig_w * self.upscale_factor
            new_h = orig_h * self.upscale_factor
            
            # Convert PIL to tensor
            to_tensor = T.ToTensor()
            img_tensor = to_tensor(image).unsqueeze(0).to(self.device)
            
            # Upscale using bicubic interpolation (high quality)
            upscaled_tensor = F.interpolate(
                img_tensor,
                size=(new_h, new_w),
                mode='bicubic',
                align_corners=False,
                antialias=True
            )
            
            # Convert back to PIL
            to_pil = T.ToPILImage()
            upscaled = to_pil(upscaled_tensor.squeeze(0).cpu())
            
            print(f"✓ Upscaled from {image.size} to {upscaled.size}")
            return upscaled
            
        except Exception as e:
            print(f"Warning: Upscaling failed: {e}")
            print("Returning original image")
            return image
    
    def batch_upscale(self, images):
        """Upscale a batch of images"""
        return [self.upscale_image(img) for img in images]

# ============================================================================
# STAGE 1: Qwen2-VL Caption
# ============================================================================

class Qwen2VLCaptioner:
    def __init__(self, config):
        self.config = config
        print("\n[2/6] Loading Qwen2-VL-2B...")
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            self.process_vision_info = process_vision_info
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.QWEN2VL_MODEL,
                cache_dir=str(config.CACHE_DIR),
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            ).eval()
            
            print(f"  ✓ Model on device: {self.model.device}")
            
            self.processor = AutoProcessor.from_pretrained(
                config.QWEN2VL_MODEL,
                cache_dir=str(config.CACHE_DIR)
            )
            
            print("✓ Qwen2-VL loaded")
            if torch.cuda.is_available():
                print(f"  ✓ GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        except Exception as e:
            print(f"✗ Error: {e}")
            raise
    
    def generate_description(self, image_path):
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "Describe this aerial image concisely in 2-3 sentences: scene type, main structures, and key features."}
                ]
            }]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=60,
                    do_sample=True, 
                    temperature=0.5,
                    top_p=0.85,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            caption = output[0].strip() if output else "Aerial view"
            # Clean up verbose phrases
            caption = caption.replace("The image shows ", "").replace("The scene shows ", "")
            caption = caption.replace("This is ", "").replace("It is ", "")
            print(f"    ✓ Caption: {caption[:70]}...")
            return caption
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return "Aerial landscape view"

# ============================================================================
# STAGE 2: Grounding DINO Detection
# ============================================================================

class GroundingDINODetector:
    def __init__(self, config):
        self.config = config
        print("\n[4/6] Loading Grounding DINO...")
        
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            self.processor = AutoProcessor.from_pretrained(
                config.GROUNDING_DINO_MODEL, cache_dir=str(config.CACHE_DIR)
            )
            # Use float32 for stable inference
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                config.GROUNDING_DINO_MODEL,
                cache_dir=str(config.CACHE_DIR),
                torch_dtype=torch.float32
            ).to(device).eval()
            
            print("✓ Grounding DINO loaded")
            print(f"  ✓ GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            self.initialized = True
        except Exception as e:
            print(f"⚠ Error: {e}")
            self.initialized = False
    
    def detect_and_count(self, image_path):
        """Returns detailed detection results"""
        if not self.initialized:
            return {"counts": {}, "summary": "", "total": 0}
        
        try:
            image = Image.open(image_path).convert('RGB')
            text_prompt = " . ".join(self.config.DETECT_CLASSES)
            
            inputs = self.processor(
                images=image, text=text_prompt, return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process detections (thresholds are now handled internally by the model)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[image.size[::-1]]
            )[0]
            
            # Filter results by confidence threshold (manual filtering after post-processing)
            box_threshold = 0.25
            if 'scores' in results:
                mask = results['scores'] >= box_threshold
                results = {
                    'boxes': results['boxes'][mask],
                    'scores': results['scores'][mask],
                    'labels': [label for i, label in enumerate(results['labels']) if mask[i]]
                }
            
            # Ensure results has proper structure even if empty
            if 'boxes' not in results or len(results['boxes']) == 0:
                results = {'boxes': [], 'scores': [], 'labels': []}
            
            # Count objects
            counts = {}
            for label in results['labels']:
                counts[label] = counts.get(label, 0) + 1
            
            total = sum(counts.values())
            
            # Create summary
            if counts:
                top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
                summary = ", ".join([f"{c} {o}" for o, c in top])
                print(f"    ✓ Detected: {summary} (total: {total})")
            else:
                summary = ""
                print(f"    ℹ No objects detected above threshold {box_threshold}")
            
            return {
                "counts": counts,
                "summary": summary,
                "total": total,
                "results": results  # Pass full results to SAM
            }
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            return {"counts": {}, "summary": "", "total": 0, "results": None}

# ============================================================================
# STAGE 3: SAM Precise Localizer
# ============================================================================

class PreciseLocalizer:
    def __init__(self, sam_model_name):
        print("\n[3/6] Loading SAM (Segment Anything Model)...")
        
        try:
            from transformers import SamModel, SamProcessor
            import torch
            
            self.processor = SamProcessor.from_pretrained(sam_model_name)
            # Use float32 for compatibility with DINO
            self.model = SamModel.from_pretrained(
                sam_model_name,
                torch_dtype=torch.float32
            ).to(device).eval()
            
            print("✓ SAM loaded successfully")
            print(f"  ✓ GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
        except Exception as e:
            print(f"Error loading SAM: {e}")
            raise
    
    def get_precise_locations(self, image, detection_results):
        """
        Get precise locations using SAM for detected objects
        
        Args:
            image: PIL Image
            detection_results: Results from Grounding DINO with bounding boxes
            
        Returns:
            Dictionary with location information and spatial descriptions
        """
        try:
            if detection_results is None or 'boxes' not in detection_results:
                return {"locations": [], "spatial_description": ""}
            
            import numpy as np
            
            boxes = detection_results['boxes'].cpu().numpy()
            labels = detection_results['labels']
            
            locations = []
            
            # Convert image to numpy for SAM
            img_array = np.array(image)
            
            # Get image dimensions for grid positioning
            img_width, img_height = image.size
            
            for idx, (box, label) in enumerate(zip(boxes, labels)):
                # Process with SAM for precise segmentation
                inputs = self.processor(
                    img_array,
                    input_boxes=[[[box.tolist()]]],
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get mask - post_process_masks is in processor, not image_processor
                masks = self.processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                    binarize=False
                )[0]
                
                # Get the best mask
                mask = masks[0, 0].numpy()
                
                # Calculate precise center from mask
                y_indices, x_indices = np.where(mask > 0.5)
                if len(x_indices) > 0:
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    
                    # Calculate 9-grid position (3x3 grid)
                    grid_x = int(center_x / (img_width / 3))
                    grid_y = int(center_y / (img_height / 3))
                    grid_x = min(grid_x, 2)  # Clamp to 0-2
                    grid_y = min(grid_y, 2)
                    
                    # Grid position names
                    grid_positions = [
                        ["top-left", "top-center", "top-right"],
                        ["middle-left", "center", "middle-right"],
                        ["bottom-left", "bottom-center", "bottom-right"]
                    ]
                    position = grid_positions[grid_y][grid_x]
                    
                    # Calculate area and relative size
                    area = np.sum(mask > 0.5)
                    relative_size = area / (img_width * img_height)
                    
                    size_desc = "large" if relative_size > 0.15 else "medium" if relative_size > 0.05 else "small"
                    
                    locations.append({
                        "object": label,
                        "position": position,
                        "center": (center_x, center_y),
                        "area": int(area),
                        "relative_size": float(relative_size),
                        "size_description": size_desc
                    })
                    
                    print(f"    ✓ {label}: {position} ({size_desc})")
            
            # Create spatial description
            if locations:
                spatial_desc = self._create_spatial_description(locations)
            else:
                spatial_desc = ""
            
            return {
                "locations": locations,
                "spatial_description": spatial_desc
            }
            
        except Exception as e:
            print(f"  ⚠ SAM localization error: {e}")
            return {"locations": [], "spatial_description": ""}
    
    def _create_spatial_description(self, locations):
        """Create a CONCISE natural language description of object locations"""
        # Group by position and count objects
        position_groups = {}
        for loc in locations:
            pos = loc['position']
            obj = loc['object']
            if pos not in position_groups:
                position_groups[pos] = {}
            position_groups[pos][obj] = position_groups[pos].get(obj, 0) + 1
        
        # Create concise descriptions with counts
        desc_parts = []
        for pos, obj_counts in position_groups.items():
            if len(obj_counts) == 1:
                obj, count = list(obj_counts.items())[0]
                if count == 1:
                    desc_parts.append(f"{obj} in {pos}")
                else:
                    desc_parts.append(f"{count} {obj}s in {pos}")
            else:
                # Multiple object types in same position
                obj_strs = [f"{count} {obj}" + ("s" if count > 1 else "") for obj, count in obj_counts.items()]
                desc_parts.append(f"{', '.join(obj_strs)} in {pos}")
        
        # Limit to 5 most important positions to keep prompt short
        return "; ".join(desc_parts[:5])

# ============================================================================
# STAGE 4: Template-based Prompt Combiner
# ============================================================================

class SmartPromptCombiner:
    """
    Template-based prompt combiner for aerial imagery
    Combines Qwen2-VL captions with Grounding DINO detections and SAM spatial positions
    More reliable than LLM-based combination for structured data
    """
    
    def __init__(self, config):
        self.config = config
        print("\n[5/6] Initializing Template-based Prompt Combiner...")
        print("✓ Using direct template-based combination (no LLM needed)")
        print("  ✓ Ensures ALL detection data (counts + spatial positions) is included")
    
    
    def combine_prompt(self, caption, detection_results):
        """
        Intelligently combine caption and detections into concise prompt
        """
        try:
            # Build structured input with ALL information from 3 models
            counts = detection_results["counts"]
            total = detection_results["total"]
            summary = detection_results.get("summary", "")
            
            counts_text = ", ".join([f"{count} {obj}" for obj, count in counts.items()])
            
            # Simple fallback if no detections
            if not counts_text or total == 0:
                combined = f"Aerial view: {caption}"
                print(f"    ✓ Combined prompt (no detections): {combined}")
                return combined
            
            # IMPROVED: Create comprehensive prompt that uses ALL model outputs
            # Include: Qwen2-VL caption + DINO detections + SAM spatial locations
            enhanced_summary = detection_results.get('summary', '')
            
            # Extract spatial layout from enhanced_summary if available
            spatial_info = ""
            if "Spatial layout:" in enhanced_summary:
                spatial_info = enhanced_summary.split("Spatial layout:")[-1].strip()
            
            # Build FULL detailed prompt using ALL model outputs
            combined = f"Satellite photograph of {caption.lower()}"
            
            # Add ALL object counts
            if counts_text:
                combined += f" with {counts_text}"
            
            # Add full spatial distribution
            if spatial_info:
                combined += f". Layout: {spatial_info}"
            elif enhanced_summary and not spatial_info:
                # Use full summary if no spatial layout extracted
                combined += f". {enhanced_summary}"
            
            # Add realism enhancers - focus on natural muted colors like real satellite imagery
            combined += ". Real satellite imagery, muted natural colors, weathered rooftops, dusty appearance, overhead view, nadir angle, natural shadows, varied building conditions, realistic urban textures, Google Maps satellite view."
            
            word_count = len(combined.split())
            print(f"    ✓ Template-based combination ({word_count} words, ~{int(word_count * 1.3)} tokens): {combined[:200]}...")
            return combined
            
        except Exception as e:
            print(f"  ⚠ Error in prompt combination: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            if detection_results.get("summary"):
                return f"Aerial view: {caption}, {detection_results['summary']}"
            return f"Aerial view: {caption}"

# ============================================================================
# STAGE 5: SDXL + ControlNet Image Generator
# ============================================================================

class SDXLControlNetGenerator:
    """
    SDXL with ControlNet for precise spatial control
    Uses edge detection from upscaled image to maintain structure
    Much better prompt adherence than text-only generation
    """
    
    def __init__(self, config):
        self.config = config
        print("\n[6/6] Loading SDXL + ControlNet...")
        
        try:
            from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
            import gc
            
            # Clear GPU memory
            print("  ℹ Clearing GPU memory for SDXL...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Load ControlNet first
            print("  ℹ Loading ControlNet (Canny edge control)...")
            self.controlnet = ControlNetModel.from_pretrained(
                config.CONTROLNET_MODEL,
                cache_dir=str(config.CACHE_DIR),
                torch_dtype=torch.float16
            )
            
            # Load SDXL with ControlNet
            print("  ℹ Loading SDXL base model...")
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                config.SDXL_MODEL,
                controlnet=self.controlnet,
                cache_dir=str(config.CACHE_DIR),
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(device)
            
            # Memory optimization
            print("  ℹ Enabling memory optimizations...")
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
            
            print("✓ SDXL + ControlNet loaded")
            print(f"  ✓ GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print("  ✓ ControlNet: Canny edge detection for spatial control")
            
        except Exception as e:
            print(f"✗ Error loading SDXL + ControlNet: {e}")
            raise
    
    def prepare_control_image(self, image):
        """
        Create canny edge map from image for ControlNet
        This preserves the structure and layout of the original image
        Uses PIL and scipy instead of cv2 to avoid libGL dependency on servers
        """
        import numpy as np
        from PIL import ImageFilter, ImageOps
        
        # Convert to grayscale
        gray = ImageOps.grayscale(image)
        
        # Apply edge detection using PIL's FIND_EDGES filter
        # Then enhance with CONTOUR for better results
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.CONTOUR)
        
        # Enhance contrast to make edges more visible
        edges = ImageOps.autocontrast(edges)
        
        # Convert back to RGB for ControlNet
        control_image = edges.convert('RGB')
        
        return control_image
    
    def generate_image(self, prompt, output_path, control_image=None):
        """
        Generate image with SDXL + ControlNet
        
        Args:
            prompt: Detailed text description
            output_path: Where to save
            control_image: Upscaled image to extract structure from
        
        ControlNet ensures the generated image follows the structure (edges)
        of the control_image while using the prompt for content/style.
        
        This gives MUCH better prompt adherence than text-only generation!
        """
        try:
            print(f"  Generating image with SDXL + ControlNet...")
            print(f"  Prompt ({len(prompt)} chars): {prompt[:150]}...")
            print(f"  Resolution: {self.config.OUTPUT_IMAGE_SIZE}x{self.config.OUTPUT_IMAGE_SIZE}")
            print(f"  Steps: {self.config.NUM_INFERENCE_STEPS}")
            
            if control_image is None:
                print("  ⚠ Warning: No control image provided, using text-only generation")
                print("     Prompt adherence will be poor. Pass upscaled image for better results.")
            
            # Prepare control image (canny edges)
            if control_image:
                print("  ℹ Extracting structure via Canny edge detection...")
                control = self.prepare_control_image(control_image)
                control = control.resize((self.config.OUTPUT_IMAGE_SIZE, self.config.OUTPUT_IMAGE_SIZE))
                
                # Save control image for debugging
                control_path = output_path.parent / f"{output_path.stem}_control.png"
                control.save(control_path)
                print(f"  ✓ Control image saved: {control_path.name}")
            else:
                control = None
            
            # Strong negative prompt to avoid oversaturated/artificial look
            negative_prompt = "cartoon, anime, illustration, drawing, painting, artistic, stylized, 3d render, cgi, digital art, oversaturated, vibrant colors, bright green, neon colors, perfect, clean, pristine, new buildings, uniform appearance, video game, miniature, tilt-shift, HDR, overprocessed"
            
            # Generate with ControlNet
            print(f"  ℹ ControlNet conditioning scale: {self.config.CONTROLNET_CONDITIONING_SCALE}")
            print(f"  ℹ Guidance scale: {self.config.GUIDANCE_SCALE}")
            
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control,  # Edge map controls structure
                num_inference_steps=self.config.NUM_INFERENCE_STEPS,
                controlnet_conditioning_scale=self.config.CONTROLNET_CONDITIONING_SCALE,
                guidance_scale=self.config.GUIDANCE_SCALE,
                height=self.config.OUTPUT_IMAGE_SIZE,
                width=self.config.OUTPUT_IMAGE_SIZE,
            ).images[0]
            
            image.save(output_path)
            print(f"  ✓ Saved: {output_path.name}")
            print(f"  ℹ Structure preserved via ControlNet edge map")
            return True
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

class EnhancedPipeline:
    def __init__(self):
        self.config = Config()
        print("\nInitializing Pipeline...")
        
        # Stage 0: Upscaler
        self.upscaler = RealESRGANUpscaler(self.config)
        
        # Stage 1: Captioner
        self.captioner = Qwen2VLCaptioner(self.config)
        
        # Stage 2: Detector
        self.detector = GroundingDINODetector(self.config)
        
        # Stage 3: SAM Localizer
        self.localizer = PreciseLocalizer(self.config.SAM_MODEL)
        
        # Stage 4: Combiner
        self.combiner = SmartPromptCombiner(self.config)
        
        # Stage 5: Image Generator (SDXL + ControlNet)
        self.generator = SDXLControlNetGenerator(self.config)
        
        print("\n" + "=" * 80)
        print("✓ Pipeline Ready!")
        print("  Stage 0: Bicubic Upscaler (4x)")
        print("  Stage 1: Qwen2-VL-2B Captioner")
        print("  Stage 2: Grounding DINO Detector")
        print("  Stage 3: SAM Precise Localizer")
        print("  Stage 4: Template-based Combiner")
        print("  Stage 5: SDXL + ControlNet Generator (1024x1024)")
        print("  Stage 6: Qwen2-VL-2B Output Description")
        print("=" * 80)
    
    def process_image(self, image_path, image_name):
        print(f"\n{'='*80}")
        print(f"Processing: {image_name}")
        print("=" * 80)
        
        try:
            from PIL import Image
            
            # Load original image
            original_image = Image.open(image_path)
            
            # Stage 0: Upscale
            print("\n[Stage 0/6] Upscaling image...")
            upscaled_image = self.upscaler.upscale_image(original_image)
            
            # Save upscaled image temporarily for processing
            upscaled_path = self.config.OUTPUT_DIR / "images" / f"{image_name}_upscaled.png"
            upscaled_image.save(upscaled_path)
            print(f"✓ Upscaled image saved: {upscaled_path}")
            
            # Stage 1: Caption (use upscaled image)
            print("\n[Stage 1/6] Generating caption with Qwen2-VL...")
            caption = self.captioner.generate_description(str(upscaled_path))
            
            # Stage 2: Detection (use upscaled image)
            print("\n[Stage 2/6] Detecting objects with Grounding DINO...")
            detection_results = self.detector.detect_and_count(str(upscaled_path))
            
            # Stage 3: Precise Localization with SAM
            print("\n[Stage 3/6] Getting precise locations with SAM...")
            location_results = self.localizer.get_precise_locations(
                upscaled_image,
                detection_results.get('results')
            )
            
            # Stage 4: Smart combination with template
            print("\n[Stage 4/6] Combining prompt with template combiner...")
            
            # Create enhanced detection summary with SAM locations
            enhanced_detection_summary = detection_results['summary']
            if location_results.get('spatial_description'):
                enhanced_detection_summary += f". Spatial layout: {location_results['spatial_description']}"
            
            # Combine using the existing combiner
            combined_prompt = self.combiner.combine_prompt(caption, {
                **detection_results,
                'summary': enhanced_detection_summary
            })
            
            # Print combined prompt
            print("\n" + "="*80)
            print("COMBINED PROMPT")
            print("="*80)
            print(combined_prompt)
            print("="*80)
            
            # Clear GPU memory before generation
            print(f"\n  ℹ Clearing GPU memory...")
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
            # Stage 5: Image Generation
            print(f"\n[Stage 5/6] Generating image with SDXL + ControlNet...")
            output_path = self.config.OUTPUT_DIR / "images" / f"{image_name}_generated.png"
            
            # Pass upscaled image as control for ControlNet
            success = self.generator.generate_image(
                combined_prompt, 
                output_path,
                control_image=upscaled_image  # Use upscaled image for structure
            )
            
            if success:
                # Stage 6: Caption the generated image
                print(f"\n[Stage 6/6] Generating description of output image...")
                generated_image = Image.open(output_path)
                
                # Generate detailed caption of the final output
                output_caption = self.captioner.generate_description(str(output_path))
                
                print(f"    ✓ Output description: {output_caption}")
                
                metadata = {
                    "source": str(image_path),
                    "output": str(output_path),
                    "output_caption": output_caption,
                    "generation_prompt": combined_prompt,
                    "models": {
                        "upscaler": self.config.UPSCALER_MODEL,
                        "caption": self.config.QWEN2VL_MODEL,
                        "detection": self.config.GROUNDING_DINO_MODEL,
                        "localizer": self.config.SAM_MODEL,
                        "combiner": "Template-based",
                        "generation": self.config.SDXL_MODEL,
                        "controlnet": self.config.CONTROLNET_MODEL
                    }
                }
                
                meta_path = self.config.OUTPUT_DIR / "metadata" / f"{image_name}.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"\n{'='*80}")
                print("✓ PROCESSING COMPLETE!")
                print("=" * 80)
                print(f"\n--- GENERATED IMAGE ---")
                print(f"Size: {generated_image.size}")
                print(f"Path: {output_path}")
                
                print(f"\n--- GENERATED IMAGE DESCRIPTION ---")
                print(output_caption)
                print("=" * 80)
                return True
            return False
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        images = list(self.config.INPUT_DIR.glob("*.jpg")) + \
                 list(self.config.INPUT_DIR.glob("*.png"))
        
        if not images:
            print(f"\n✗ No images in {self.config.INPUT_DIR}")
            return
        
        if self.config.MAX_IMAGES:
            images = images[:self.config.MAX_IMAGES]
        
        print(f"\n{'='*80}")
        print(f"Processing {len(images)} image(s)")
        print(f"{'='*80}")
        
        successful = 0
        for img_path in images:
            if self.process_image(img_path, img_path.stem):
                successful += 1
        
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Successful: {successful}/{len(images)}")
        print(f"Output: {self.config.OUTPUT_DIR}")
        print("=" * 80)

def main():
    print("\n" + "=" * 80)
    print("ENHANCED PIPELINE - SERVER GPU MODE")
    print("=" * 80)
    print("\nPipeline Features:")
    print("  ✓ PyTorch Bicubic 4x upscaling")
    print("  ✓ Qwen2-VL-2B for RS image captioning")
    print("  ✓ Grounding DINO for zero-shot object detection")
    print("  ✓ SAM for precise 9-grid object localization")
    print("  ✓ Template-based prompt combination")
    print("  ✓ SDXL + ControlNet (1024x1024, edge-guided generation)")
    print("\nServer Requirements:")
    print("  • GPU: 24GB+ VRAM (A10G/A100/H100 recommended)")
    print("  • RAM: 32GB+ system memory")
    print("  • Storage: 100GB+ (for models + cache + outputs)")
    print("=" * 80)
    
    try:
        pipeline = EnhancedPipeline()
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final GPU cleanup
        if torch.cuda.is_available():
            print("\n  ℹ Final GPU memory cleanup...")
            torch.cuda.empty_cache()
            print("  ✓ Cleanup complete")

if __name__ == "__main__":
    main()
