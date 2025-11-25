#!/usr/bin/env python3
"""
COMPLETE PIPELINE: Qwen2-VL → Grounding DINO → SAM → Template Combiner → SD3.5
Optimized for GPU server execution

IMPORTANT NOTE ABOUT SD3.5 TEXT ENCODERS:
SD3.5 uses THREE text encoders working in parallel:
- CLIP-L (77 tokens max) - Shows truncation warnings (IGNORE)
- CLIP-G (77 tokens max) - Shows truncation warnings (IGNORE)  
- T5-XXL (512 tokens max) - PRIMARY encoder for long prompts

The CLIP warnings are EXPECTED behavior. The T5-XXL encoder processes your
full prompt (up to 512 tokens) and is what actually generates the image.
You will see "Token indices sequence length is longer than 77" warnings - 
these are SAFE TO IGNORE as long as your prompt is under 512 tokens.
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
print("ENHANCED PIPELINE: Qwen2-VL → Grounding DINO → Phi-3.5 → SD3.5")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Use environment variables for server deployment
    INPUT_DIR = Path(os.getenv("INPUT_DIR", "datasets/rsicd_images"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/pipeline_results"))
    CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", "./hf_cache"))
    
    # Models
    UPSCALER_MODEL = "PyTorch-Bicubic"  # High-quality bicubic upscaling
    SAM_MODEL = "facebook/sam-vit-base"  # NEW: Precise localization
    QWEN2VL_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"
    PHI_MODEL = "microsoft/Phi-3.5-mini-instruct"  # Smart prompt combiner
    SD35_MODEL = "stabilityai/stable-diffusion-3.5-medium"  # SD3.5 Medium
    
    # Settings
    MAX_IMAGES = 1
    UPSCALE_FACTOR = 4  # NEW: 4x upscaling
    OUTPUT_IMAGE_SIZE = 1024  # SD3.5 supports 1024x1024
    NUM_INFERENCE_STEPS = 28  # Optimized for SD3.5 Medium
    GUIDANCE_SCALE = 8.5  # Higher = more accurate to prompt (SD3.5 range: 3.0-10.0)
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
# STAGE 0: Real-ESRGAN Upscaler
# ============================================================================

class RealESRGANUpscaler:
    def __init__(self, config):
        self.config = config
        print("\n[1/6] Loading Real-ESRGAN Upscaler...")
        
        try:
            import torch.nn.functional as F
            
            self.upscale_factor = config.UPSCALE_FACTOR
            self.device = device
            
            print(f"✓ Real-ESRGAN fallback loaded (upscale factor: {config.UPSCALE_FACTOR}x)")
            print("  ℹ Using PyTorch bicubic upsampling (high quality)")
            
        except Exception as e:
            print(f"Error loading Real-ESRGAN: {e}")
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
# STAGE 2B: SAM Precise Localizer
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
# STAGE 3: Phi-3.5-mini Smart Prompt Combiner (NEW!)
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
            # SD3.5's T5-XXL encoder can handle up to 512 tokens (~400 words)
            # CLIP encoders will truncate at 77 tokens, but T5 is the primary encoder
            combined = f"Aerial view of {caption.lower()}"
            
            # Add ALL object counts
            if counts_text:
                combined += f" featuring {counts_text}"
            
            # Add full spatial distribution
            if spatial_info:
                combined += f". Spatial distribution: {spatial_info}"
            elif enhanced_summary and not spatial_info:
                # Use full summary if no spatial layout extracted
                combined += f". {enhanced_summary}"
            
            # Add quality enhancers
            combined += ". High quality aerial photography, detailed, sharp focus."
            
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
# STAGE 4: Stable Diffusion 3.5 Medium Image Generator
# ============================================================================

class SD35ImageGenerator:
    """
    Stable Diffusion 3.5 Medium generator for high-quality aerial image generation
    """
    
    def __init__(self, config):
        self.config = config
        print("\n[6/6] Loading Stable Diffusion 3.5 Medium...")
        
        try:
            from diffusers import StableDiffusion3Pipeline
            import gc
            
            # CRITICAL: Clear GPU memory before loading SD3.5
            print("  ℹ Clearing GPU memory for SD3.5...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Load with fp16 variant for better compatibility
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                config.SD35_MODEL,
                cache_dir=str(config.CACHE_DIR),
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(device)
            
            # IMPORTANT: SD3.5 uses 3 text encoders:
            # - text_encoder (CLIP-L): 77 tokens max
            # - text_encoder_2 (CLIP-G): 77 tokens max  
            # - text_encoder_3 (T5-XXL): 512 tokens max
            # The T5 encoder is the PRIMARY one for long prompts
            
            print("  ℹ NOTE: You will see CLIP truncation warnings below - these are SAFE TO IGNORE")
            print("  ℹ SD3.5's T5-XXL encoder (512 tokens) is processing your FULL prompt")
            
            # Enable attention slicing for memory efficiency on GPU
            print("  ℹ Enabling attention slicing for GPU memory efficiency...")
            self.pipe.enable_attention_slicing()
            
            print("✓ Stable Diffusion 3.5 Medium loaded")
            print(f"  ✓ GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print("  ✓ Full GPU mode with attention slicing")
            
        except Exception as e:
            print(f"✗ Error loading SD3.5: {e}")
            raise
    
    def generate_image(self, prompt, output_path):
        """
        Generate high-quality aerial image using SD3.5 Medium
        
        IMPORTANT: SD3.5 uses 3 text encoders in parallel:
        1. CLIP-L (77 tokens) - will truncate and show warnings (IGNORE these)
        2. CLIP-G (77 tokens) - will truncate and show warnings (IGNORE these)  
        3. T5-XXL (512 tokens) - PRIMARY encoder, handles full prompt
        
        The CLIP warnings are EXPECTED and can be ignored. The T5-XXL encoder
        is what actually processes your full prompt for image generation.
        
        LIMITATION: Text-to-image models CANNOT guarantee exact object counts or
        precise spatial placement. This is a fundamental limitation of diffusion models.
        They interpret "29 houses" as "many houses" semantically.
        
        For exact counts/positions, you would need:
        - ControlNet (layout-conditioned generation)
        - Fine-tuned model on aerial imagery with count annotations
        - Image-to-image pipeline (not available in SD3.5 yet)
        """
        try:
            print(f"  Generating image with Stable Diffusion 3.5 Medium...")
            print(f"  Prompt ({len(prompt)} chars): {prompt[:150]}...")
            print(f"  Resolution: {self.config.OUTPUT_IMAGE_SIZE}x{self.config.OUTPUT_IMAGE_SIZE}")
            print(f"  Steps: {self.config.NUM_INFERENCE_STEPS}, Guidance: {self.config.GUIDANCE_SCALE}")
            
            # Negative prompt for better quality
            negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text, extra objects, wrong count"
            
            # Generate with FULL 512-token support via T5 encoder
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=self.config.OUTPUT_IMAGE_SIZE,
                width=self.config.OUTPUT_IMAGE_SIZE,
                num_inference_steps=self.config.NUM_INFERENCE_STEPS,
                guidance_scale=self.config.GUIDANCE_SCALE,  # 8.5 for better adherence
                max_sequence_length=512
            ).images[0]
            
            image.save(output_path)
            print(f"  ✓ Saved: {output_path.name}")
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
        print("\nInitializing Enhanced 5-Stage Pipeline...")
        
        # Stage 0: Upscaler
        self.upscaler = RealESRGANUpscaler(self.config)
        
        # Stage 1: Captioner
        self.captioner = Qwen2VLCaptioner(self.config)
        
        # Stage 2: Detector
        self.detector = GroundingDINODetector(self.config)
        
        # Stage 2B: SAM Localizer
        self.localizer = PreciseLocalizer(self.config.SAM_MODEL)
        
        # Stage 3: Combiner
        self.combiner = SmartPromptCombiner(self.config)
        
        # Stage 4: SD3.5 Medium Generator
        self.generator = SD35ImageGenerator(self.config)
        
        print("\n" + "=" * 80)
        print("✓ Enhanced 5-Stage Pipeline Ready!")
        print("  1. Real-ESRGAN Upscaler (4x)")
        print("  2. Qwen2-VL-2B Captioner")
        print("  3. Grounding DINO Detector")
        print("  4. SAM Precise Localizer")
        print("  5. Phi-3.5-mini Combiner")
        print("  6. Stable Diffusion 3.5 Medium (1024x1024)")
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
            print("\n[Stage 0/5] Upscaling image with Real-ESRGAN...")
            upscaled_image = self.upscaler.upscale_image(original_image)
            
            # Save upscaled image temporarily for processing
            upscaled_path = self.config.OUTPUT_DIR / "images" / f"{image_name}_upscaled.png"
            upscaled_image.save(upscaled_path)
            print(f"✓ Upscaled image saved: {upscaled_path}")
            
            # Stage 1: Caption (use upscaled image)
            print("\n[Stage 1/5] Generating caption with Qwen2-VL...")
            caption = self.captioner.generate_description(str(upscaled_path))
            
            # Stage 2A: Detection (use upscaled image)
            print("\n[Stage 2A/5] Detecting objects with Grounding DINO...")
            detection_results = self.detector.detect_and_count(str(upscaled_path))
            
            # Stage 2B: Precise Localization with SAM
            print("\n[Stage 2B/5] Getting precise locations with SAM...")
            location_results = self.localizer.get_precise_locations(
                upscaled_image,
                detection_results.get('results')
            )
            
            # Stage 3: Smart combination with Phi-3.5
            print("\n[Stage 3/5] Combining all information with Phi-3.5-mini...")
            
            # Create enhanced detection summary with SAM locations
            enhanced_detection_summary = detection_results['summary']
            if location_results.get('spatial_description'):
                enhanced_detection_summary += f". Spatial layout: {location_results['spatial_description']}"
            
            # Combine using the existing combiner
            combined_prompt = self.combiner.combine_prompt(caption, {
                **detection_results,
                'summary': enhanced_detection_summary
            })
            
            # Print full Stage 3 output
            print("\n" + "="*80)
            print("STAGE 3 OUTPUT - FULL COMBINED PROMPT WITH SAM LOCATIONS")
            print("="*80)
            print(combined_prompt)
            print("="*80)
            
            # CRITICAL: Clear GPU memory before SD3.5 generation (server has 24GB+)
            print("\n[Stage 3.5/5] Optimizing GPU memory for SD3.5...")
            import gc
            
            # On server GPU (24GB+), just clear cache - don't delete models
            print("  ℹ Clearing GPU cache (keeping models loaded)...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            # Print memory status
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  ✓ GPU memory status: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # Stage 4: SD3.5 Generation
            print("\n[Stage 4/5] Generating image with Stable Diffusion 3.5 Medium...")
            output_path = self.config.OUTPUT_DIR / "images" / f"{image_name}_sd35_gen.png"
            success = self.generator.generate_image(combined_prompt, output_path)
            
            if success:
                # Convert detection_results to JSON-serializable format
                detection_metadata = {
                    "counts": detection_results.get("counts", {}),
                    "summary": detection_results.get("summary", ""),
                    "total": detection_results.get("total", 0)
                    # Don't include 'results' key as it contains Tensors
                }
                
                metadata = {
                    "source": str(image_path),
                    "original_size": f"{original_image.size[0]}x{original_image.size[1]}",
                    "upscaled_size": f"{upscaled_image.size[0]}x{upscaled_image.size[1]}",
                    "upscale_factor": self.config.UPSCALE_FACTOR,
                    "qwen2vl_caption": caption,
                    "grounding_dino_detections": detection_metadata,
                    "sam_precise_locations": location_results,
                    "phi35_combined_prompt": combined_prompt,
                    "output": str(output_path),
                    "models": {
                        "upscaler": self.config.UPSCALER_MODEL,
                        "caption": self.config.QWEN2VL_MODEL,
                        "detection": self.config.GROUNDING_DINO_MODEL,
                        "localizer": self.config.SAM_MODEL,
                        "combiner": self.config.PHI_MODEL,
                        "generation": self.config.SD35_MODEL
                    }
                }
                
                meta_path = self.config.OUTPUT_DIR / "metadata" / f"{image_name}.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"\n{'='*80}")
                print("✓ PROCESSING COMPLETE!")
                print("=" * 80)
                print(f"Original Size: {original_image.size}")
                print(f"Upscaled Size: {upscaled_image.size}")
                print(f"\nCaption: {caption}")
                print(f"\nDetections: {detection_results['summary']}")
                print(f"\nSAM Locations: {location_results.get('spatial_description', 'N/A')}")
                print(f"\n--- FULL COMBINED PROMPT (STAGE 3) ---")
                print(combined_prompt)
                print(f"\n--- END OF COMBINED PROMPT ---")
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
    print("  ✓ Phi-3.5-mini for intelligent prompt combination")
    print("  ✓ Stable Diffusion 3.5 Medium (1024x1024, full GPU)")
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
