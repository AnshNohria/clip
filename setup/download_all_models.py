#!/usr/bin/env python3
"""
Pre-download ALL HuggingFace models used by the RemoteCLIP pipeline.

Saves each repo under:
    <repo_root>/models/<org>__<name>/

No GPU required — this only downloads weights/config/tokenizer files.

Uses HF_TOKEN from the repo .env for gated repos (FLUX.1-dev, SD3.5 Large).

Usage (from the clip repo root):
    python setup/download_all_models.py
    python setup/download_all_models.py --skip-gated
    python setup/download_all_models.py --only qwen,flux

Gated models (accept the license on HuggingFace with the SAME account as HF_TOKEN):
    https://huggingface.co/black-forest-labs/FLUX.1-dev
    https://huggingface.co/stabilityai/stable-diffusion-3.5-large
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ENV_CANDIDATES = [
    REPO_ROOT / ".env",
    Path(__file__).resolve().parent / ".env",
]


def _load_dotenv() -> None:
    """Load repo .env so HF_TOKEN is available (override any empty shell vars)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("WARNING: python-dotenv not installed; relying on process env only")
        return
    for env_path in ENV_CANDIDATES:
        if env_path.exists():
            # override=True so a blank HF_TOKEN in the shell cannot block .env
            load_dotenv(env_path, override=True)
            print(f"Loaded env from {env_path}")
            return
    print(f"WARNING: no .env found (looked in {[str(p) for p in ENV_CANDIDATES]})")


def _resolve_hf_token() -> Optional[str]:
    """Read HF_TOKEN / HUGGING_FACE_HUB_TOKEN and strip quotes/whitespace."""
    raw = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or ""
    )
    token = raw.strip().strip('"').strip("'").strip()
    return token or None


def _mask_token(token: str) -> str:
    if not token:
        return "(none)"
    if len(token) <= 10:
        return "***"
    return f"{token[:6]}...{token[-4:]}"


def _local_dir_for(repo_id: str, models_dir: Optional[Path] = None) -> Path:
    """Map 'org/name' -> models/org__name (Windows-safe, flat folder)."""
    root = models_dir if models_dir is not None else MODELS_DIR
    return root / repo_id.replace("/", "__")


# ---------------------------------------------------------------------------
# Model catalog (kept in sync with remoteclip_pipeline/config.py)
# ---------------------------------------------------------------------------

MODELS: List[Dict] = [
    # --- Synthetic generation pipeline (A100 x2 build) ---
    {
        "key": "qwen",
        "name": "Qwen2.5-VL-7B-Instruct",
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "gated": False,
        "notes": "Scene analysis + multi-caption generation (analysis GPU)",
    },
    {
        "key": "gdino",
        "name": "Grounding DINO Base",
        "repo_id": "IDEA-Research/grounding-dino-base",
        "gated": False,
        "notes": "Object / layout detection for prompts + captions",
    },
    {
        "key": "flux",
        "name": "FLUX.1-dev",
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "gated": True,
        "notes": "Primary image generator (gen GPU). ~24GB download.",
        "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
    },
    {
        "key": "sd35",
        "name": "Stable Diffusion 3.5 Large",
        "repo_id": "stabilityai/stable-diffusion-3.5-large",
        "gated": True,
        "notes": "Optional SD3.5 fallback (set sd_backend='sd3' in config).",
        "license_url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
    },
    # --- Zoom-crops pipeline (still used by Stage 1B) ---
    {
        "key": "sam",
        "name": "SAM ViT-Huge",
        "repo_id": "facebook/sam-vit-huge",
        "gated": False,
        "notes": "Used by zoom_crops_pipeline.py for object masks",
    },
    # --- LoRA training backbone ---
    {
        "key": "remoteclip",
        "name": "RemoteCLIP-ViT-B-32",
        "repo_id": "chendelong/RemoteCLIP",
        "gated": False,
        "notes": "RemoteCLIP backbone checkpoint used by Stage 2/3 trainers",
        # Only the ViT-B-32 weight is needed; skip the larger ViT-L/H variants.
        "allow_patterns": ["RemoteCLIP-ViT-B-32.pt", "README.md", "*.json", "*.txt"],
    },
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _login_hf(token: Optional[str], require_for_gated: bool) -> str:
    """
    Authenticate with HuggingFace using HF_TOKEN from .env.

    Injects the token into the process environment so every
    huggingface_hub / transformers / diffusers call picks it up, and
    verifies it with whoami() before any gated download starts.
    """
    if not token:
        msg = (
            "HF_TOKEN not found. Put it in the repo .env as:\n"
            "  HF_TOKEN=hf_xxxxxxxx\n"
            "(no spaces; quotes optional). Gated repos (FLUX / SD3.5) require it."
        )
        if require_for_gated:
            print(f"ERROR: {msg}")
            sys.exit(1)
        print(f"WARNING: {msg}")
        return ""

    # Make sure every HF client (hub, transformers, diffusers) sees the token
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token

    try:
        from huggingface_hub import HfApi, login

        login(token=token, add_to_git_credential=False)
        info = HfApi(token=token).whoami(token=token)
        username = info.get("name") or info.get("fullname") or "unknown"
        print(f"Logged in to HuggingFace as '{username}' (token {_mask_token(token)})")
        return token
    except Exception as e:
        print(f"ERROR: HuggingFace login / whoami failed: {e}")
        print(
            "Check that HF_TOKEN in .env is a valid token and that you have "
            "accepted gated model licenses on huggingface.co with the same account."
        )
        sys.exit(1)


def _download_one(
    model: Dict,
    token: Optional[str],
    force: bool,
    models_dir: Path,
) -> bool:
    """Download one model repo into models/<org>__<name>/. Returns success."""
    from huggingface_hub import snapshot_download

    repo_id = model["repo_id"]
    local_dir = _local_dir_for(repo_id, models_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Repo:      {repo_id}")
    print(f"  Local dir: {local_dir}")
    if model.get("notes"):
        print(f"  Notes:     {model['notes']}")
    if model.get("gated"):
        print(f"  License:   {model.get('license_url', 'see HuggingFace page')}")
        print(f"  Auth:      using HF_TOKEN={_mask_token(token or '')}")

    # Skip if already present and not forcing (heuristic: any weight file exists)
    if not force and _looks_downloaded(local_dir):
        print("  SKIP: already present (use --force to re-download)")
        return True

    if model.get("gated") and not token:
        print("  FAIL: gated repo requires HF_TOKEN")
        return False

    try:
        kwargs = {
            "repo_id": repo_id,
            "local_dir": str(local_dir),
            # Explicit token on every call — required for gated repos.
            # token=True falls back to the cached/login token if string is empty.
            "token": token if token else True,
            "resume_download": True,
            "max_workers": 8,
        }
        if model.get("allow_patterns"):
            kwargs["allow_patterns"] = model["allow_patterns"]

        snapshot_download(**kwargs)
        print(f"  OK: {model['name']}")
        return True
    except Exception as e:
        print(f"  FAIL: {model['name']}: {e}")
        if model.get("gated"):
            print(
                "  Hint: open the license URL above, click "
                "'Agree and access repository', then re-run. "
                "Your HF_TOKEN must belong to the same account."
            )
        return False


def _looks_downloaded(local_dir: Path) -> bool:
    """True if the folder already contains weight-like files."""
    if not local_dir.exists():
        return False
    weight_globs = ("*.safetensors", "*.bin", "*.pt", "*.ckpt", "*.msgpack")
    for pattern in weight_globs:
        if any(local_dir.rglob(pattern)):
            return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all RemoteCLIP pipeline models into ./models/"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(MODELS_DIR),
        help=f"Destination root (default: {MODELS_DIR})",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Comma-separated keys to download. "
            f"Available: {', '.join(m['key'] for m in MODELS)}"
        ),
    )
    parser.add_argument(
        "--skip-gated",
        action="store_true",
        help="Skip gated models (FLUX.1-dev, SD3.5 Large)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if weight files already exist locally",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_dotenv()

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    selected = MODELS
    if args.only:
        keys = {k.strip().lower() for k in args.only.split(",") if k.strip()}
        unknown = keys - {m["key"] for m in MODELS}
        if unknown:
            print(f"ERROR: unknown model keys: {sorted(unknown)}")
            print(f"Available: {[m['key'] for m in MODELS]}")
            return 1
        selected = [m for m in MODELS if m["key"] in keys]

    if args.skip_gated:
        selected = [m for m in selected if not m.get("gated")]

    needs_gated = any(m.get("gated") for m in selected)
    token = _login_hf(_resolve_hf_token(), require_for_gated=needs_gated)

    print()
    print("=" * 70)
    print("REMOTECLIP MODEL DOWNLOADER")
    print("=" * 70)
    print(f"Destination: {models_dir}")
    print(f"HF auth:     {'yes (' + _mask_token(token) + ')' if token else 'no'}")
    print(f"Models:      {len(selected)}")
    for m in selected:
        tag = " [GATED]" if m.get("gated") else ""
        print(f"  - {m['key']}: {m['repo_id']}{tag}")
    print("=" * 70)
    print()

    results: Dict[str, bool] = {}
    for i, model in enumerate(selected, 1):
        print("-" * 70)
        print(f"[{i}/{len(selected)}] {model['name']}")
        print("-" * 70)
        ok = _download_one(
            model,
            token=token or None,
            force=args.force,
            models_dir=models_dir,
        )
        results[model["key"]] = ok
        print()

    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    ok_count = sum(1 for v in results.values() if v)
    for model in selected:
        status = "OK" if results.get(model["key"]) else "FAILED"
        print(f"  [{status}] {model['key']} -> {_local_dir_for(model['repo_id'], models_dir)}")
    print()
    print(f"{ok_count}/{len(results)} succeeded")
    print(f"Models root: {models_dir}")
    print("=" * 70)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
