# Stage 1 & Stage 2: Synthetic-Heavy CLIP Fine-Tuning Pipeline

This document describes the two-stage training pipeline for fine-tuning RemoteCLIP on aerial/satellite imagery with a synthetic-heavy data mix.

## Overview

### Stage 1: Build and Freeze the Augmented + Synthetic Dataset
Creates a high-quality training corpus from:
- **Real Augmented Set**: Original images + geometric/photometric augmentations + M2B/B2C caption enrichment + VLM captions + zoom crops
- **Synthetic Set**: Images generated via the 5-stage pipeline (RealESRGAN → Qwen2-VL → Grounding DINO → SAM → SD3.5)
- **Quality Filtering**: CLIP/RemoteCLIP similarity scoring + sanity checks

### Stage 2: Synthetic-Heavy LoRA Fine-Tuning
Trains LoRA adapters on frozen RemoteCLIP:
- **Data Mix**: 70-90% synthetic + 10-30% real
- **Losses**: CLIP contrastive + rare class weighting + optional fine-grained word-region alignment
- **Output**: LoRA adapter weights + projection heads

## File Structure

```
clip/
├── stage1_dataset_builder.py    # Stage 1: Dataset building
├── quality_filter.py            # Quality filtering with CLIP/RemoteCLIP
├── stage2_lora_finetune.py      # Stage 2: LoRA fine-tuning
├── requirements_hpc.txt         # Updated dependencies
├── run_hpc_pipeline.py          # Existing 5-stage synthetic pipeline
│
├── datasets/
│   ├── rsicd_images/            # Source real images
│   ├── stage1_corpus/           # Stage 1 output
│   │   ├── dataset.json         # Master index
│   │   ├── images/
│   │   │   ├── real_aug/        # Real augmented images
│   │   │   └── synthetic/       # Synthetic images
│   │   ├── metadata/            # Per-image JSON
│   │   └── filtered/            # Quality-filtered output
│   │       ├── filtered_dataset.json
│   │       └── quality_report.json
│
└── checkpoints/
    └── stage2_lora/             # Stage 2 output
        ├── lora_final.pt        # LoRA adapter weights
        ├── projection_heads_final.pt
        └── training_log.json
```

## Quick Start

### Prerequisites
```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements_hpc.txt
```

### Stage 1: Build Dataset
```bash
python stage1_dataset_builder.py
```

This will:
1. Load images from `datasets/rsicd_images/`
2. Generate M2B+B2C enriched captions (algorithmic)
3. Optionally generate Qwen2-VL VLM captions (requires GPU)
4. Create augmented versions (flip, rotate, crop, color jitter)
5. Generate zoom crops from detected regions
6. Save to `datasets/stage1_corpus/`

### Quality Filtering
```bash
python quality_filter.py
```

This will:
1. Score each pair with CLIP/RemoteCLIP similarity
2. Run sanity checks (object counts, spatial layout)
3. Analyze image quality (brightness, contrast, blur)
4. Keep top 70% by composite score
5. Save to `datasets/stage1_corpus/filtered/`

### Stage 2: LoRA Fine-Tuning
```bash
python stage2_lora_finetune.py
```

This will:
1. Load RemoteCLIP backbone (frozen)
2. Apply LoRA adapters to attention + MLP layers
3. Create trainable projection heads
4. Train with synthetic-heavy data mix
5. Save checkpoints to `checkpoints/stage2_lora/`

## Configuration

### Stage 1 (`stage1_dataset_builder.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_AUGMENTATIONS_PER_IMAGE` | 5 | Augmented versions per real image |
| `ENABLE_ZOOM_CROPS` | True | Generate object-level crops |
| `USE_VLM_CAPTIONS` | True | Use Qwen2-VL (requires GPU) |
| `USE_M2B_B2C` | True | Use algorithmic enrichment |

### Quality Filter (`quality_filter.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLIP_WEIGHT` | 0.50 | Weight for CLIP similarity |
| `SANITY_WEIGHT` | 0.25 | Weight for sanity checks |
| `IMAGE_QUALITY_WEIGHT` | 0.25 | Weight for image quality |
| `MIN_CLIP_SCORE_SYNTHETIC` | 0.25 | Minimum CLIP score for synthetic |
| `KEEP_TOP_PERCENT` | 0.70 | Keep top 70% by score |

### Stage 2 (`stage2_lora_finetune.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SYNTHETIC_RATIO` | 0.80 | Proportion of synthetic data |
| `LORA_RANK` | 8 | LoRA rank (r) |
| `LORA_ALPHA` | 16.0 | LoRA scaling factor |
| `LEARNING_RATE` | 1e-4 | Learning rate |
| `NUM_EPOCHS` | 10 | Training epochs |
| `USE_FINE_GRAINED_LOSS` | True | Enable word-region alignment |
| `FINE_GRAINED_WEIGHT` | 0.3 | Weight for fine-grained loss |
| `RARE_CLASS_WEIGHT` | 2.0 | Upweight factor for rare classes |

## Caption Enrichment: M2B + B2C

### Mask-to-Box (M2B)
Extracts regions using:
1. Edge detection (Sobel/Laplacian)
2. 3x3 grid spatial analysis
3. Color analysis (green→vegetation, blue→water, etc.)

### Box-to-Caption (B2C)
Generates descriptions from regions:
1. Scene type classification (urban, vegetation, water, etc.)
2. Spatial position phrases ("in the upper left", "at the center")
3. Element descriptions based on color analysis

## Fine-Grained Alignment Loss

When enabled (`USE_FINE_GRAINED_LOSS=True`), the model learns to align:
- Word embeddings from captions
- Spatial region features from images

This improves localization of objects mentioned in captions.

## Output Formats

### dataset.json
```json
{
  "name": "stage1_corpus",
  "stats": {
    "real_original": 26,
    "real_augmented": 130,
    "zoom_crops": 52,
    "synthetic": 0
  },
  "pairs": [
    {
      "image_id": "abc123...",
      "image_path": "images/real_aug/abc123.jpg",
      "caption": "Aerial view showing...",
      "source": "real",
      "caption_sources": ["m2b_b2c", "qwen2vl"]
    }
  ]
}
```

### training_log.json
```json
{
  "config": {
    "model_type": "RemoteCLIP",
    "lora_rank": 8,
    "synthetic_ratio": 0.8
  },
  "training_log": [
    {
      "step": 50,
      "epoch": 1,
      "total_loss": 2.345,
      "acc_i2t": 0.125,
      "acc_t2i": 0.115
    }
  ]
}
```

## GPU Requirements

| Stage | GPU Memory | Notes |
|-------|------------|-------|
| Stage 1 (no VLM) | 0 GB | CPU-only |
| Stage 1 (with VLM) | ~6 GB | Qwen2-VL-2B |
| Quality Filter | ~4 GB | CLIP inference |
| Stage 2 | ~8-12 GB | LoRA training |

## Troubleshooting

### "No CLIP model available"
```bash
pip install open_clip_torch
# or
pip install git+https://github.com/openai/CLIP.git
```

### "RemoteCLIP not found"
The pipeline will automatically download from HuggingFace, or:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('chendelong/RemoteCLIP', 'RemoteCLIP-ViT-B-32.pt')"
```

### "CUDA out of memory"
Reduce batch size in `Stage2Config`:
```python
BATCH_SIZE: int = 16  # Default is 32
```

## References

- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) - Remote sensing CLIP
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
- [CLIP](https://openai.com/research/clip) - Contrastive Language-Image Pre-training
