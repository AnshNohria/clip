"""
Download arampacha/rsicd into ./Datasets with images + captions.
Layout:
  Datasets/rsicd/
    train/images/*.jpg
    test/images/*.jpg
    valid/images/*.jpg
    train.jsonl
    test.jsonl
    valid.jsonl
    metadata.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency. Run: uv pip install datasets pillow")
        sys.exit(1)

    root = Path(__file__).resolve().parent / "Datasets" / "rsicd"
    root.mkdir(parents=True, exist_ok=True)

    print("Downloading arampacha/rsicd (uses HF cache if already present)...")
    ds = load_dataset("arampacha/rsicd")

    # Discover column names once
    sample = ds["train"][0]
    cols = list(sample.keys())
    print("Columns:", cols)

    image_key = "image" if "image" in sample else next(
        (k for k in cols if "image" in k.lower()), None
    )
    caption_key = next(
        (k for k in ("captions", "caption", "text", "sentences") if k in sample),
        None,
    )
    if image_key is None or caption_key is None:
        print("Could not find image/caption columns. Got:", cols)
        print("Sample:", {k: type(v).__name__ for k, v in sample.items()})
        sys.exit(1)

    print(f"Using image_key={image_key!r}, caption_key={caption_key!r}")
    summary = {"dataset": "arampacha/rsicd", "splits": {}}

    for split in ds.keys():
        split_ds = ds[split]
        img_dir = root / split / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        records = []

        n = len(split_ds)
        print(f"\nExporting {split}: {n} examples -> {img_dir}")

        for i in range(n):
            ex = split_ds[i]
            img = ex[image_key]
            if hasattr(img, "convert"):
                img = img.convert("RGB")

            filename = f"{i:05d}.jpg"
            rel_path = f"{split}/images/{filename}"
            img.save(img_dir / filename, format="JPEG", quality=95)

            caps = ex[caption_key]
            if isinstance(caps, str):
                caps = [caps]
            else:
                caps = [str(c) for c in list(caps)]

            records.append(
                {
                    "id": i,
                    "image": rel_path,
                    "captions": caps,
                    "caption": caps[0] if caps else "",
                }
            )

            if (i + 1) % 500 == 0 or (i + 1) == n:
                print(f"  {split}: {i + 1}/{n}")

        jsonl_path = root / f"{split}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        json_path = root / f"{split}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        summary["splits"][split] = {
            "num_examples": n,
            "images_dir": str((root / split / "images").relative_to(root.parent.parent)),
            "jsonl": str(jsonl_path.relative_to(root.parent.parent)),
        }
        print(f"  wrote {jsonl_path.name} and {json_path.name}")

    meta_path = root / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Dataset root: {root}")
    print("Structure:")
    print("  Datasets/rsicd/train/images/*.jpg")
    print("  Datasets/rsicd/test/images/*.jpg")
    print("  Datasets/rsicd/valid/images/*.jpg")
    print("  Datasets/rsicd/train.jsonl  (image path + captions)")
    print("  Datasets/rsicd/test.jsonl")
    print("  Datasets/rsicd/valid.jsonl")


if __name__ == "__main__":
    main()