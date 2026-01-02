# Model Download Instructions

## Problem
Loading all models simultaneously during first run requires too much VRAM (~15GB+) because downloading models uses more memory than just loading cached models.

## Solution
Pre-download models separately before running the main pipeline.

---

## Option 1: Download Only SD 3.5 (Quick)

If other models are already cached, download just SD 3.5:

```bash
# On remote machine
source /home/jovyan/clip/clip-venv/bin/activate
python download_sd35.py
```

This downloads ~10GB of SD 3.5 model files.

---

## Option 2: Download All Models (Recommended for First Time)

Download all models sequentially to avoid VRAM issues:

```bash
# On remote machine
source /home/jovyan/clip/clip-venv/bin/activate
python download_all_models.py
```

This will download:
1. **Qwen2-VL-7B-Instruct** (~15GB) - Scene analysis
2. **Grounding DINO** (~700MB) - Object detection
3. **SAM ViT-Huge** (~2.5GB) - Segmentation
4. **Stable Diffusion 3.5 Medium** (~10GB) - Image generation
5. **CLIP ViT-B-32** (~300MB) - Quality scoring

**Total: ~28GB download**

Each model is downloaded and then cleared from memory before the next one.

---

## After Download

Once models are cached, run the main pipeline:

```bash
python run_stage1.py
```

Models will load much faster from cache and use less VRAM during initialization.

---

## Cache Location

Models are cached at: `/home/jovyan/clip/checkpoints/huggingface/`

Check your `.env` file for `HF_HOME` setting.

---

## Troubleshooting

**If download still fails:**
1. Restart your Jupyter kernel/session to free all GPU memory
2. Check free VRAM: `nvidia-smi`
3. Make sure no other processes are using GPU
4. Run download script when GPU has at least 8GB free

**SD 3.5 specific issues:**
- Download requires ~6-8GB free VRAM
- Loading from cache only needs ~2-3GB
- If stuck at "Loading checkpoint shards", restart and try download script

---

## How It Works

**During Download:**
- Model files are downloaded from HuggingFace Hub
- Model is temporarily loaded to verify download
- Model is immediately deleted from VRAM
- Cache is cleared before next model

**During Pipeline Run:**
- Models load from local cache (fast)
- Sequential CPU offloading keeps VRAM usage low
- All models stay loaded in CPU/GPU memory for fast access
