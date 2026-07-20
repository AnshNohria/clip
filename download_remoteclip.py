"""
Download RemoteCLIP ViT-B/32 for local/offline research use.

Output:
    models/RemoteCLIP/RemoteCLIP-ViT-B-32.pt

Run:
    uv pip install huggingface_hub
    uv run download_remoteclip.py

Optional authenticated download:
    # Linux/macOS
    export HF_TOKEN=hf_your_token

    # Windows PowerShell
    $env:HF_TOKEN = "hf_your_token"

Then run the script normally.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path


REPO_ID = "chendelong/RemoteCLIP"
FILENAME = "RemoteCLIP-ViT-B-32.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Missing dependency. Install it with:")
        print("  uv pip install huggingface_hub")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "models" / "RemoteCLIP"
    output_path = output_dir / FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # If present, preserve it and avoid another network request.
    if output_path.is_file() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {output_path}")
        print(f"Size: {size_mb:.2f} MB")
        print(f"SHA-256: {sha256(output_path)}")
        print("Ready for offline use.")
        return

    token = os.getenv("HF_TOKEN")
    print(f"Downloading {REPO_ID}/{FILENAME}")
    print(f"Destination: {output_path}")

    try:
        cached_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="model",
            token=token,
        )
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        sys.exit(1)

    cached_path = Path(cached_file)

    # Copy, rather than symlink, so the final model stays usable even if
    # you later clear Hugging Face's cache.
    shutil.copy2(cached_path, output_path)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        print("Download/copy did not produce a valid model file.")
        sys.exit(1)

    print("\nDownload complete.")
    print(f"Local model: {output_path.resolve()}")
    print(f"Size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"SHA-256: {sha256(output_path)}")
    print("\nFor offline use, load this exact local path:")
    print('  model_path = "models/RemoteCLIP/RemoteCLIP-ViT-B-32.pt"')


if __name__ == "__main__":
    main()