# Quick Setup Guide for Cluster

## 1. Clone Repository on Cluster

```bash
git clone https://github.com/AnshNohria/clip.git
cd clip
```

## 2. Install Dependencies

```bash
# Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
pip install -r requirements_hpc.txt
```

## 3. Setup Environment

```bash
# Set HuggingFace token (replace with your actual token)
export HF_TOKEN=your_huggingface_token_here

# Create directories
mkdir -p datasets/rsicd_images
mkdir -p outputs/pipeline_results
mkdir -p hf_cache

# Copy your images to datasets/rsicd_images/
```

## 4. Run Pipeline

```bash
python run_hpc_pipeline.py
```

## Models Downloaded (first run):
- Qwen2-VL-2B (~4.4GB)
- Grounding DINO (~2GB)
- SAM (~1GB)
- Phi-3.5-mini (~2.5GB)
- SD3.5 Medium (~10GB)

Total: ~20GB models will be downloaded and cached.
