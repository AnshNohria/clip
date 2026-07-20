#!/usr/bin/env python3
"""
RemoteCLIP Retrieval Evaluation: Original vs M2B/B2C Captions
============================================================
Loads the local RemoteCLIP ViT-B/32 checkpoint and measures text-image
retrieval on the RSICD train / valid / test splits.

Two caption variants are compared:
  1) original  -> the human-authored `caption` field from Datasets/rsicd/<split>.jsonl
  2) m2b_b2c   -> the 5 rule-based captions per image produced by the M2B/B2C
                  pipeline, read from Datasets/enriched_caption/<split>_rich.jsonl
                  ("captions" list). Evaluated with the standard multi-caption
                  retrieval protocol (5 texts per image).

Metrics (both directions):
  - Text -> Image  (T2I): R@1, R@5, R@10, median rank, mean rank, MRR
  - Image -> Text  (I2T): R@1, R@5, R@10, median rank, mean rank, MRR

For m2b_b2c, T2I ranks every one of the 5 captions against all images; I2T
takes, per image, the best rank among its own 5 captions.

Expected layout:
  models/RemoteCLIP/RemoteCLIP-ViT-B-32.pt
  Datasets/rsicd/{train,valid,test}/images/...
  Datasets/rsicd/{train,valid,test}.jsonl                (original captions)
  Datasets/enriched_caption/{train,valid,test}_rich.jsonl (M2B/B2C captions)

Run:
  python evaluate_remoteclip_retrieval.py --splits test
  python evaluate_remoteclip_retrieval.py --splits test valid
  python evaluate_remoteclip_retrieval.py --variants m2b_b2c
  python evaluate_remoteclip_retrieval.py --batch-size 64 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================================
# PATHS / DEFAULTS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "RemoteCLIP" / "RemoteCLIP-ViT-B-32.pt"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "Datasets" / "rsicd"
DEFAULT_ENRICHED_DIR = PROJECT_ROOT / "Datasets" / "enriched_caption"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "retrieval_eval"
SPLITS = ("train", "valid", "test")
VARIANTS = ("original", "m2b_b2c")
RECALL_KS = (1, 5, 10)


# ============================================================================
# DATA
# ============================================================================

@dataclass
class Sample:
    image_rel: str
    image_path: Path
    caption: str            # original human caption (may be "")
    m2b_captions: List[str]  # M2B/B2C captions (may be empty)
    sample_id: int


def _resolve_image_path(rel: str, dataset_dir: Path, split: str) -> Optional[Path]:
    img_path = dataset_dir / rel
    if img_path.is_file():
        return img_path
    alt = dataset_dir / split / "images" / Path(rel).name
    return alt if alt.is_file() else None


def load_split_samples(
    split: str,
    dataset_dir: Path,
    enriched_dir: Path,
) -> List[Sample]:
    """
    Build samples for one split, driven by the M2B/B2C rich jsonl
    (Datasets/enriched_caption/<split>_rich.jsonl). Original human captions
    are pulled from Datasets/rsicd/<split>.jsonl, matched by image path.
    """
    rich_jsonl = enriched_dir / f"{split}_rich.jsonl"
    orig_jsonl = dataset_dir / f"{split}.jsonl"

    if not rich_jsonl.exists():
        raise FileNotFoundError(
            f"Missing M2B/B2C rich jsonl for split '{split}': {rich_jsonl}\n"
            f"Generate it with:\n"
            f"  python caption_enrichment.py --splits {split} --rich"
        )

    # Map image_rel -> original caption
    orig_by_image: Dict[str, str] = {}
    if orig_jsonl.exists():
        with open(orig_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rel = row.get("image", "")
                cap = (row.get("caption") or "").strip()
                if rel and cap:
                    orig_by_image[rel] = cap
    else:
        print(f"  [warn] original captions jsonl not found: {orig_jsonl}")

    samples: List[Sample] = []
    with open(rich_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rel = row.get("image", "")
            if not rel:
                continue

            img_path = _resolve_image_path(rel, dataset_dir, split)
            if img_path is None:
                print(f"  [warn] missing image (line {line_no}): {rel}")
                continue

            m2b_caps = [c.strip() for c in (row.get("captions") or []) if c and c.strip()]
            caption = orig_by_image.get(rel) or (row.get("original_caption") or "").strip()

            samples.append(
                Sample(
                    image_rel=rel,
                    image_path=img_path,
                    caption=caption,
                    m2b_captions=m2b_caps,
                    sample_id=int(row.get("id", len(samples))),
                )
            )

    if not samples:
        raise RuntimeError(f"No usable samples found for split '{split}' in {rich_jsonl}")

    return samples


class ImagePathDataset(Dataset):
    """Lazy image loader for RemoteCLIP preprocessing."""

    def __init__(self, paths: List[Path], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            tensor = self.preprocess(image)
        except Exception as exc:
            print(f"  [warn] failed to load {path}: {exc}")
            tensor = torch.zeros(3, 224, 224)
        return tensor, idx


# ============================================================================
# MODEL
# ============================================================================

def load_remoteclip(model_path: Path, device: str):
    """
    Load RemoteCLIP ViT-B/32 from a local .pt checkpoint via open_clip.

    Official RemoteCLIP pattern:
      model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
      model.load_state_dict(torch.load(ckpt))
    """
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open-clip-torch is required. Install with: pip install open-clip-torch"
        ) from exc

    if not model_path.is_file():
        raise FileNotFoundError(
            f"RemoteCLIP checkpoint not found: {model_path}\n"
            f"Run: python download_remoteclip.py"
        )

    print(f"Loading RemoteCLIP from: {model_path}")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    ckpt = torch.load(model_path, map_location="cpu")
    # RemoteCLIP hubs sometimes wrap weights; accept raw state_dict or nested.
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    print(f"  Device: {device}")
    return model, preprocess, tokenizer


# ============================================================================
# ENCODING
# ============================================================================

@torch.no_grad()
def encode_images(
    model,
    preprocess,
    image_paths: List[Path],
    device: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> torch.Tensor:
    """Return L2-normalized image embeddings [N, D]."""
    dataset = ImagePathDataset(image_paths, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    feats = [None] * len(image_paths)
    for images, indices in tqdm(loader, desc="  Encoding images", leave=False):
        images = images.to(device, non_blocking=True)
        emb = model.encode_image(images)
        emb = F.normalize(emb.float(), dim=-1)
        for i, idx in enumerate(indices.tolist()):
            feats[idx] = emb[i].cpu()

    return torch.stack(feats, dim=0)


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Return L2-normalized text embeddings [N, D]."""
    all_feats = []
    for start in tqdm(range(0, len(texts), batch_size), desc="  Encoding texts", leave=False):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(batch).to(device)
        emb = model.encode_text(tokens)
        emb = F.normalize(emb.float(), dim=-1)
        all_feats.append(emb.cpu())
    return torch.cat(all_feats, dim=0)


# ============================================================================
# METRICS
# ============================================================================

def summarize_ranks(ranks: np.ndarray, ks: Tuple[int, ...] = RECALL_KS) -> Dict[str, float]:
    ranks = ranks.astype(np.float64)
    metrics = {}
    for k in ks:
        metrics[f"R@{k}"] = float(np.mean(ranks <= k))
    metrics["median_rank"] = float(np.median(ranks))
    metrics["mean_rank"] = float(np.mean(ranks))
    metrics["mean_reciprocal_rank"] = float(np.mean(1.0 / ranks))
    return metrics


@torch.no_grad()
def evaluate_retrieval(
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    text2img: List[int],
    chunk_size: int = 512,
) -> Dict[str, Dict[str, float]]:
    """
    General multi-caption retrieval.

    image_feats: [N, D] L2-normalized image embeddings.
    text_feats:  [M, D] L2-normalized text embeddings.
    text2img:    length-M list mapping each text to its image index (0..N-1).

    Reduces to standard 1:1 retrieval when M == N and text2img == range(N).

    T2I: for each text, rank of its paired image among all N images.
    I2T: for each image, the best (smallest) rank among its own texts.
    """
    n = image_feats.shape[0]
    m = text_feats.shape[0]
    t2i_map = torch.as_tensor(text2img, dtype=torch.long)

    # ---- Text -> Image ----
    t2i_ranks = np.empty(m, dtype=np.int64)
    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        sim = text_feats[start:end] @ image_feats.T  # [b, N]
        idx = torch.arange(start, end)
        pos_sim = sim[torch.arange(end - start), t2i_map[idx]].unsqueeze(1)
        # rank = 1 + number of images scoring strictly higher than the positive
        t2i_ranks[start:end] = (
            (sim > pos_sim).sum(dim=1).cpu().numpy() + 1
        )

    # ---- Image -> Text (best rank among the image's own captions) ----
    from collections import defaultdict

    img2texts: Dict[int, List[int]] = defaultdict(list)
    for j, i in enumerate(text2img):
        img2texts[i].append(j)

    i2t_ranks = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = image_feats[start:end] @ text_feats.T  # [b, M]
        for r in range(end - start):
            img_idx = start + r
            positives = img2texts.get(img_idx)
            if not positives:
                i2t_ranks[img_idx] = m
                continue
            row = sim[r]
            best_pos = row[positives].max()
            i2t_ranks[img_idx] = int((row > best_pos).sum().item()) + 1

    return {
        "text_to_image": summarize_ranks(t2i_ranks),
        "image_to_text": summarize_ranks(i2t_ranks),
    }


# ============================================================================
# EVALUATION ORCHESTRATION
# ============================================================================

def _variant_texts(
    samples: List[Sample], variant: str
) -> Tuple[List[str], List[int]]:
    """
    Return (texts, text2img) for a variant.
    - original: one caption per image (text2img = image index).
    - m2b_b2c : all M2B/B2C captions per image, each mapped to its image.
    Images with no caption for the variant are skipped for that variant.
    """
    texts: List[str] = []
    text2img: List[int] = []
    if variant == "original":
        for i, s in enumerate(samples):
            if s.caption:
                texts.append(s.caption)
                text2img.append(i)
    elif variant == "m2b_b2c":
        for i, s in enumerate(samples):
            for cap in s.m2b_captions:
                texts.append(cap)
                text2img.append(i)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return texts, text2img


def evaluate_split(
    split: str,
    samples: List[Sample],
    model,
    preprocess,
    tokenizer,
    device: str,
    batch_size: int,
    num_workers: int,
    variants: Tuple[str, ...] = VARIANTS,
) -> Dict:
    print(f"\n{'=' * 60}")
    print(f"Split: {split}  ({len(samples)} images)")
    print(f"{'=' * 60}")

    image_paths = [s.image_path for s in samples]

    # Show a sample caption per variant for sanity.
    for variant in variants:
        ex_texts, _ = _variant_texts(samples[:1], variant)
        example = ex_texts[0] if ex_texts else ""
        print(f"  ex[{variant:<9}]: {example[:110]}")

    # Images are shared across variants — encode once
    image_feats = encode_images(
        model, preprocess, image_paths, device, batch_size, num_workers
    )

    results = {"split": split, "num_samples": len(samples), "caption_variants": {}}

    for variant in variants:
        texts, text2img = _variant_texts(samples, variant)
        if not texts:
            print(f"\n  Caption variant: {variant} -> no texts, skipping")
            continue
        print(f"\n  Caption variant: {variant}  ({len(texts)} texts)")
        text_feats = encode_texts(model, tokenizer, texts, device, batch_size=batch_size * 2)
        metrics = evaluate_retrieval(image_feats, text_feats, text2img)
        results["caption_variants"][variant] = metrics

        t2i, i2t = metrics["text_to_image"], metrics["image_to_text"]
        print(
            f"    T2I  R@1={t2i['R@1']:.4f}  R@5={t2i['R@5']:.4f}  "
            f"R@10={t2i['R@10']:.4f}  medR={t2i['median_rank']:.1f}  MRR={t2i['mean_reciprocal_rank']:.4f}"
        )
        print(
            f"    I2T  R@1={i2t['R@1']:.4f}  R@5={i2t['R@5']:.4f}  "
            f"R@10={i2t['R@10']:.4f}  medR={i2t['median_rank']:.1f}  MRR={i2t['mean_reciprocal_rank']:.4f}"
        )

    # Compact deltas vs the `original` baseline for T2I
    if "original" in results["caption_variants"]:
        base = results["caption_variants"]["original"]["text_to_image"]
        deltas = {}
        for variant, metrics in results["caption_variants"].items():
            if variant == "original":
                continue
            t2i = metrics["text_to_image"]
            deltas[variant] = {
                k: float(t2i[k] - base[k])
                for k in ("R@1", "R@5", "R@10", "mean_reciprocal_rank")
            }
        results["delta_vs_original_t2i"] = deltas
        for variant, d in deltas.items():
            print(f"\n  Delta ({variant} - original) T2I:")
            for k, v in d.items():
                print(f"    {k}: {v:+.4f}")

    return results


def print_summary_table(all_results: List[Dict]) -> None:
    print(f"\n{'#' * 88}")
    print("SUMMARY - RemoteCLIP retrieval (original vs m2b_b2c)")
    print(f"{'#' * 88}")
    header = (
        f"{'Split':<8} {'Variant':<10} "
        f"{'T2I R@1':>9} {'T2I R@5':>9} {'T2I R@10':>9} {'T2I MRR':>9} "
        f"{'I2T R@1':>9} {'I2T R@5':>9} {'I2T R@10':>9}"
    )
    print(header)
    print("-" * len(header))

    for res in all_results:
        split = res["split"]
        for variant, metrics in res["caption_variants"].items():
            t2i, i2t = metrics["text_to_image"], metrics["image_to_text"]
            print(
                f"{split:<8} {variant:<10} "
                f"{t2i['R@1']:9.4f} {t2i['R@5']:9.4f} {t2i['R@10']:9.4f} "
                f"{t2i['mean_reciprocal_rank']:9.4f} "
                f"{i2t['R@1']:9.4f} {i2t['R@5']:9.4f} {i2t['R@10']:9.4f}"
            )
    print(f"{'#' * 88}\n")


# ============================================================================
# MAIN
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate RemoteCLIP retrieval: original vs enriched captions"
    )
    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to RemoteCLIP-ViT-B-32.pt",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="RSICD root with <split>/images and optional <split>.jsonl",
    )
    p.add_argument(
        "--enriched-dir",
        type=Path,
        default=DEFAULT_ENRICHED_DIR,
        help="Directory with enriched <split>.jsonl files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write metrics JSON",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Which splits to evaluate (default: train valid test)",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        choices=list(VARIANTS),
        help="Which caption variants to evaluate (default: original m2b_b2c)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per split (for smoke tests)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RemoteCLIP Retrieval Evaluation")
    print("=" * 72)
    print(f"Model     : {args.model}")
    print(f"Dataset   : {args.dataset_dir}")
    print(f"Enriched  : {args.enriched_dir}")
    print(f"Splits    : {', '.join(args.splits)}")
    print(f"Variants  : {', '.join(args.variants)}")
    print(f"Device    : {args.device}")
    print(f"Output    : {args.output_dir}")

    model, preprocess, tokenizer = load_remoteclip(args.model, args.device)

    all_results: List[Dict] = []
    t0 = time.time()

    for split in args.splits:
        samples = load_split_samples(split, args.dataset_dir, args.enriched_dir)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
            print(f"  Using first {len(samples)} samples (--max-samples)")

        split_results = evaluate_split(
            split=split,
            samples=samples,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            variants=tuple(args.variants),
        )
        all_results.append(split_results)

        # Persist per-split immediately (crash-safe)
        split_out = args.output_dir / f"retrieval_{split}.json"
        with open(split_out, "w", encoding="utf-8") as f:
            json.dump(split_results, f, indent=2)
        print(f"  Saved: {split_out}")

    print_summary_table(all_results)

    summary = {
        "model": str(args.model),
        "dataset_dir": str(args.dataset_dir),
        "enriched_dir": str(args.enriched_dir),
        "device": args.device,
        "elapsed_sec": time.time() - t0,
        "splits": all_results,
    }
    summary_path = args.output_dir / "retrieval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Full summary written to: {summary_path}")
    print(f"Elapsed: {summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    # Windows consoles are often cp1252; keep unicode status prints safe.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
