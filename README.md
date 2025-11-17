# Enhanced RS Image Processing Pipeline

Production-ready pipeline for remote sensing image processing using state-of-the-art models.

## Features

- **PyTorch Bicubic Upscaling**: 4x image upscaling
- **Qwen2-VL-2B**: Advanced RS image captioning
- **Grounding DINO**: Zero-shot object detection
- **SAM**: Precise 9-grid object localization
- **Phi-3.5-mini**: Intelligent prompt combination
- **Stable Diffusion 3.5 Medium**: High-quality image generation (1024x1024)

## Server Requirements

- **GPU**: 24GB+ VRAM (A10G/A100/H100 recommended)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ (for models + cache + outputs)
- **CUDA**: 12.4+

## Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd clip
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
pip install -r requirements_hpc.txt
```

### 3. Set Environment Variables

```bash
# Required: HuggingFace token for model access
export HF_TOKEN=your_huggingface_token_here

# Optional: Configure paths (defaults shown)
export INPUT_DIR=datasets/rsicd_images
export OUTPUT_DIR=outputs/pipeline_results
export HF_CACHE_DIR=./hf_cache

# Recommended: PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
```

### 4. Prepare Dataset

Place your input images in the configured input directory:

```bash
mkdir -p datasets/rsicd_images
# Copy your images here
```

## Usage

### Run Pipeline

```bash
python run_hpc_pipeline.py
```

### Pipeline Stages

1. **Stage 0**: 4x upscaling with PyTorch bicubic interpolation
2. **Stage 1**: Image captioning with Qwen2-VL-2B
3. **Stage 2A**: Object detection with Grounding DINO
4. **Stage 2B**: Precise localization with SAM (9-grid positioning)
5. **Stage 3**: Prompt combination with Phi-3.5-mini
6. **Stage 4**: Image generation with SD3.5 Medium

### Output Structure

```
outputs/pipeline_results/
├── images/
│   ├── image_name_upscaled.png
│   └── image_name_sd35_gen.png
└── metadata/
    └── image_name.json
```

## Configuration

Edit the `Config` class in `run_hpc_pipeline.py` for custom settings:

```python
class Config:
    # Paths
    INPUT_DIR = Path(os.getenv("INPUT_DIR", "datasets/rsicd_images"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/pipeline_results"))
    CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", "./hf_cache"))
    
    # Model parameters
    UPSCALE_FACTOR = 4
    OUTPUT_IMAGE_SIZE = 1024
    MAX_FINAL_PROMPT_WORDS = 50
```

## Troubleshooting

### GPU Memory Issues

- Ensure you have 24GB+ VRAM
- Check: `nvidia-smi`
- Adjust: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512`

### Model Download Issues

- Verify HF_TOKEN is set correctly
- Check internet connectivity
- Models will be cached in HF_CACHE_DIR for reuse

### No CUDA Error

The pipeline requires CUDA. It will exit with error if GPU is not available.

## License

See LICENSE file for details.

## Citation

If you use this pipeline, please cite the respective model papers:
- Qwen2-VL
- Grounding DINO
- Segment Anything Model (SAM)
- Phi-3.5
- Stable Diffusion 3.5
