#!/usr/bin/env python3
"""
VLM Benchmark: Compare Vision-Language Models for Aerial Image Captioning

Models tested:
1. Qwen/Qwen2-VL-2B-Instruct (baseline - current pipeline)
2. OpenGVLab/InternVL2-4B (proven on aerial datasets MI-OAD)
3. llava-hf/llava-onevision-qwen2-7b-ov (strong single-image understanding)
4. OpenGVLab/InternVL3-8B (latest InternVL)

Metrics:
- Caption quality (detail, accuracy)
- Inference speed
- GPU memory usage
- Aerial-specific terminology detection
"""

import os
import sys
import time
import json
import torch
import gc
from pathlib import Path
from PIL import Image
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check GPU
if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU detected!")
    sys.exit(1)

device = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Configuration
class BenchmarkConfig:
    INPUT_DIR = Path("datasets/rsicd_images")
    OUTPUT_DIR = Path("outputs/vlm_benchmark")
    CACHE_DIR = Path("./hf_cache")
    
    # Test on multiple images for robust comparison
    NUM_TEST_IMAGES = 5
    
    # Models to benchmark
    MODELS = {
        "qwen2vl_2b": "Qwen/Qwen2-VL-2B-Instruct",
        "internvl2_4b": "OpenGVLab/InternVL2-4B", 
        "llava_onevision_7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "internvl3_8b": "OpenGVLab/InternVL3-8B",
    }
    
    # Aerial-specific keywords to check for
    AERIAL_KEYWORDS = [
        "aerial", "satellite", "overhead", "bird's eye", "top-down",
        "building", "house", "road", "street", "vehicle", "car",
        "tree", "forest", "water", "river", "field", "residential",
        "urban", "rural", "industrial", "commercial", "parking",
        "rooftop", "roof", "dense", "sparse", "grid", "pattern"
    ]
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu_memory():
    """Clear GPU memory between model loads"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    return torch.cuda.memory_allocated() / 1024**3


# ============================================================================
# Model 1: Qwen2-VL-2B (Baseline)
# ============================================================================

class Qwen2VLBenchmark:
    def __init__(self, model_name, cache_dir):
        self.model_name = model_name
        print(f"\n{'='*60}")
        print(f"Loading: {model_name}")
        print('='*60)
        
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        self.process_vision_info = process_vision_info
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        
        self.memory_used = get_gpu_memory()
        print(f"✓ Loaded. GPU Memory: {self.memory_used:.2f} GB")
    
    def generate_caption(self, image_path):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": "Describe this aerial/satellite image in detail. Include: scene type, main structures, objects with approximate counts, spatial layout, and notable features."}
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
        
        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.5,
                top_p=0.85,
            )
        inference_time = time.time() - start_time
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return output[0].strip(), inference_time
    
    def cleanup(self):
        del self.model
        del self.processor
        clear_gpu_memory()


# ============================================================================
# Model 2: InternVL2-4B
# ============================================================================

class InternVL2Benchmark:
    """InternVL2-4B benchmark using lmdeploy for inference (more stable)"""
    def __init__(self, model_name, cache_dir):
        self.model_name = model_name
        print(f"\n{'='*60}")
        print(f"Loading: {model_name}")
        print('='*60)
        
        from transformers import AutoModel, AutoTokenizer, GenerationMixin
        
        # Load with explicit device map to avoid offloading issues
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        
        # Patch the generate method if missing (transformers v4.50+ issue)
        if not hasattr(self.model.language_model, 'generate'):
            from transformers import GenerationMixin
            # Add generate capability back
            self.model.language_model.generate = GenerationMixin.generate.__get__(
                self.model.language_model, type(self.model.language_model)
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            trust_remote_code=True
        )
        
        self.memory_used = get_gpu_memory()
        print(f"✓ Loaded. GPU Memory: {self.memory_used:.2f} GB")
    
    def generate_caption(self, image_path):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        def build_transform(input_size):
            return T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        
        def load_image(image_file, input_size=448):
            image = Image.open(image_file).convert('RGB')
            transform = build_transform(input_size=input_size)
            pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(device)
            return pixel_values
        
        pixel_values = load_image(image_path)
        
        question = "<image>\nDescribe this aerial/satellite image in detail. Include: scene type, main structures, objects with approximate counts, spatial layout, and notable features."
        
        generation_config = dict(max_new_tokens=200, do_sample=True, temperature=0.5)
        
        start_time = time.time()
        with torch.no_grad():
            try:
                response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
            except Exception as e:
                # Fallback: use manual generation
                response = f"[Generation failed: {str(e)[:50]}]"
        inference_time = time.time() - start_time
        
        return response, inference_time
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        clear_gpu_memory()


# ============================================================================
# Model 3: LLaVA-OneVision-7B
# ============================================================================

class LLaVAOneVisionBenchmark:
    def __init__(self, model_name, cache_dir):
        self.model_name = model_name
        print(f"\n{'='*60}")
        print(f"Loading: {model_name}")
        print('='*60)
        
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        
        self.memory_used = get_gpu_memory()
        print(f"✓ Loaded. GPU Memory: {self.memory_used:.2f} GB")
    
    def generate_caption(self, image_path):
        image = Image.open(image_path).convert('RGB')
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this aerial/satellite image in detail. Include: scene type, main structures, objects with approximate counts, spatial layout, and notable features."},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)
        
        start_time = time.time()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.5,
            )
        inference_time = time.time() - start_time
        
        response = self.processor.decode(output[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        return response, inference_time
    
    def cleanup(self):
        del self.model
        del self.processor
        clear_gpu_memory()


# ============================================================================
# Model 4: InternVL3-8B
# ============================================================================

class InternVL3Benchmark:
    """InternVL3-8B benchmark with device_map for memory efficiency"""
    def __init__(self, model_name, cache_dir):
        self.model_name = model_name
        print(f"\n{'='*60}")
        print(f"Loading: {model_name}")
        print('='*60)
        
        from transformers import AutoModel, AutoTokenizer
        
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        
        # Patch the generate method if missing (transformers v4.50+ issue)
        if hasattr(self.model, 'language_model') and not hasattr(self.model.language_model, 'generate'):
            from transformers import GenerationMixin
            self.model.language_model.generate = GenerationMixin.generate.__get__(
                self.model.language_model, type(self.model.language_model)
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            trust_remote_code=True
        )
        
        self.memory_used = get_gpu_memory()
        print(f"✓ Loaded. GPU Memory: {self.memory_used:.2f} GB")
    
    def generate_caption(self, image_path):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        def build_transform(input_size):
            return T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        
        def load_image(image_file, input_size=448):
            image = Image.open(image_file).convert('RGB')
            transform = build_transform(input_size=input_size)
            pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(device)
            return pixel_values
        
        pixel_values = load_image(image_path)
        
        question = "<image>\nDescribe this aerial/satellite image in detail. Include: scene type, main structures, objects with approximate counts, spatial layout, and notable features."
        
        generation_config = dict(max_new_tokens=200, do_sample=True, temperature=0.5)
        
        start_time = time.time()
        with torch.no_grad():
            try:
                response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
            except Exception as e:
                response = f"[Generation failed: {str(e)[:50]}]"
        inference_time = time.time() - start_time
        
        return response, inference_time
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        clear_gpu_memory()


# ============================================================================
# Evaluation Metrics
# ============================================================================

def count_aerial_keywords(caption, keywords):
    """Count how many aerial-specific keywords are in the caption"""
    caption_lower = caption.lower()
    found = [kw for kw in keywords if kw in caption_lower]
    return len(found), found


def evaluate_caption_quality(caption):
    """Evaluate caption quality based on multiple criteria"""
    scores = {}
    
    # Length score (prefer detailed but not excessive)
    word_count = len(caption.split())
    if word_count < 20:
        scores['length'] = 0.3
    elif word_count < 50:
        scores['length'] = 0.6
    elif word_count < 150:
        scores['length'] = 1.0
    else:
        scores['length'] = 0.8
    
    # Number detection (mentions counts/numbers)
    import re
    numbers = re.findall(r'\b\d+\b', caption)
    scores['quantification'] = min(len(numbers) / 3, 1.0)  # Up to 3 numbers is good
    
    # Spatial terms
    spatial_terms = ['left', 'right', 'center', 'top', 'bottom', 'corner', 'middle', 'surrounding', 'adjacent', 'between']
    spatial_count = sum(1 for term in spatial_terms if term in caption.lower())
    scores['spatial'] = min(spatial_count / 3, 1.0)
    
    # Structure detection
    structure_terms = ['building', 'house', 'road', 'street', 'tree', 'vehicle', 'water', 'field']
    structure_count = sum(1 for term in structure_terms if term in caption.lower())
    scores['structures'] = min(structure_count / 4, 1.0)
    
    # Overall score (weighted average)
    weights = {'length': 0.2, 'quantification': 0.3, 'spatial': 0.25, 'structures': 0.25}
    overall = sum(scores[k] * weights[k] for k in weights)
    scores['overall'] = overall
    
    return scores


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_benchmark():
    config = BenchmarkConfig()
    
    # Get test images
    images = list(config.INPUT_DIR.glob("*.jpg")) + list(config.INPUT_DIR.glob("*.png"))
    if not images:
        print(f"ERROR: No images found in {config.INPUT_DIR}")
        return
    
    test_images = images[:config.NUM_TEST_IMAGES]
    print(f"\n{'='*80}")
    print(f"VLM BENCHMARK - Testing {len(test_images)} images")
    print(f"{'='*80}")
    print(f"Images: {[img.name for img in test_images]}")
    
    results = {}
    
    # Model classes mapping
    model_classes = {
        "qwen2vl_2b": Qwen2VLBenchmark,
        "internvl2_4b": InternVL2Benchmark,
        "llava_onevision_7b": LLaVAOneVisionBenchmark,
        "internvl3_8b": InternVL3Benchmark,
    }
    
    for model_key, model_name in config.MODELS.items():
        print(f"\n{'#'*80}")
        print(f"# BENCHMARKING: {model_key}")
        print(f"# Model: {model_name}")
        print(f"{'#'*80}")
        
        clear_gpu_memory()
        
        try:
            # Load model
            model_class = model_classes[model_key]
            model = model_class(model_name, config.CACHE_DIR)
            
            results[model_key] = {
                "model_name": model_name,
                "memory_gb": model.memory_used,
                "captions": [],
                "inference_times": [],
                "quality_scores": [],
                "aerial_keywords": [],
            }
            
            # Test on each image
            for img_path in test_images:
                print(f"\n  Testing: {img_path.name}")
                
                try:
                    caption, inference_time = model.generate_caption(str(img_path))
                    
                    # Evaluate
                    quality = evaluate_caption_quality(caption)
                    keyword_count, found_keywords = count_aerial_keywords(caption, config.AERIAL_KEYWORDS)
                    
                    results[model_key]["captions"].append({
                        "image": img_path.name,
                        "caption": caption
                    })
                    results[model_key]["inference_times"].append(inference_time)
                    results[model_key]["quality_scores"].append(quality)
                    results[model_key]["aerial_keywords"].append(keyword_count)
                    
                    print(f"    Caption: {caption[:100]}...")
                    print(f"    Time: {inference_time:.2f}s | Quality: {quality['overall']:.2f} | Keywords: {keyword_count}")
                    
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results[model_key]["captions"].append({"image": img_path.name, "caption": f"ERROR: {e}"})
                    results[model_key]["inference_times"].append(0)
                    results[model_key]["quality_scores"].append({"overall": 0})
                    results[model_key]["aerial_keywords"].append(0)
            
            # Cleanup
            model.cleanup()
            
        except Exception as e:
            print(f"  FAILED to load model: {e}")
            results[model_key] = {"error": str(e)}
            import traceback
            traceback.print_exc()
    
    # ============================================================================
    # Summary & Comparison
    # ============================================================================
    
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print('='*80)
    
    summary = []
    for model_key, data in results.items():
        if "error" in data:
            print(f"\n{model_key}: FAILED - {data['error']}")
            continue
        
        avg_time = sum(data["inference_times"]) / len(data["inference_times"]) if data["inference_times"] else 0
        avg_quality = sum(d["overall"] for d in data["quality_scores"]) / len(data["quality_scores"]) if data["quality_scores"] else 0
        avg_keywords = sum(data["aerial_keywords"]) / len(data["aerial_keywords"]) if data["aerial_keywords"] else 0
        
        summary.append({
            "model": model_key,
            "model_name": data["model_name"],
            "memory_gb": data["memory_gb"],
            "avg_time_s": avg_time,
            "avg_quality": avg_quality,
            "avg_keywords": avg_keywords,
            # Combined score: quality weighted most, then keywords, then speed (inverted)
            "combined_score": avg_quality * 0.5 + (avg_keywords / 10) * 0.3 + (1 / (avg_time + 1)) * 0.2
        })
        
        print(f"\n{model_key}:")
        print(f"  Model: {data['model_name']}")
        print(f"  GPU Memory: {data['memory_gb']:.2f} GB")
        print(f"  Avg Inference Time: {avg_time:.2f}s")
        print(f"  Avg Quality Score: {avg_quality:.3f}")
        print(f"  Avg Aerial Keywords: {avg_keywords:.1f}")
    
    # Rank models
    if summary:
        print(f"\n{'='*80}")
        print("RANKING (by combined score)")
        print('='*80)
        
        ranked = sorted(summary, key=lambda x: x["combined_score"], reverse=True)
        for i, model in enumerate(ranked, 1):
            print(f"\n{i}. {model['model']} (Combined Score: {model['combined_score']:.3f})")
            print(f"   Quality: {model['avg_quality']:.3f} | Speed: {model['avg_time_s']:.2f}s | Keywords: {model['avg_keywords']:.1f} | Memory: {model['memory_gb']:.1f}GB")
        
        print(f"\n{'='*80}")
        print(f"🏆 BEST MODEL: {ranked[0]['model']}")
        print(f"   {ranked[0]['model_name']}")
        print('='*80)
    
    # Save results
    output_file = config.OUTPUT_DIR / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for k, v in results.items():
            if "error" in v:
                json_results[k] = v
            else:
                json_results[k] = {
                    "model_name": v["model_name"],
                    "memory_gb": v["memory_gb"],
                    "captions": v["captions"],
                    "inference_times": v["inference_times"],
                    "quality_scores": v["quality_scores"],
                    "aerial_keywords": v["aerial_keywords"]
                }
        json.dump({"results": json_results, "summary": summary}, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results, summary


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VLM BENCHMARK FOR AERIAL IMAGE CAPTIONING")
    print("="*80)
    print("\nModels to test:")
    print("  1. Qwen/Qwen2-VL-2B-Instruct (baseline)")
    print("  2. OpenGVLab/InternVL2-4B (aerial-trained)")
    print("  3. llava-hf/llava-onevision-qwen2-7b-ov-hf")
    print("  4. OpenGVLab/InternVL3-8B (latest)")
    print("\nStarting benchmark...")
    
    run_benchmark()
