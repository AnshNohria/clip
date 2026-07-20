#!/usr/bin/env python3
"""
RemoteCLIP-style M2B (Mask-to-Box) + B2C (Box-to-Caption)
=======================================================
Paper-standard pipeline for converting remote-sensing images into
OpenCLIP-compatible image-caption pairs.

Pipeline:
  RGB image
    -> PseudoMaskGenerator (classical CV land-cover mask; no ML)
    -> M2B (findContours per class + boundingRect)
    -> boxes + labels
    -> B2C (5 rule-based captions: center / peripheral / random subsets)
    -> JSONL {image, caption}

RSICD has no ground-truth masks, so masks are synthesized from color
land-cover classification + morphology. Adjacent same-class regions
merge into one box (connected-component limit noted in the paper).

Depends only on: opencv-python-headless, numpy, Pillow (no ML).

Alternates for M2B when you already have instance masks:
  - torchvision.ops.masks_to_boxes
  - supervision.mask_to_xyxy
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Tunables for pseudo-mask synthesis, M2B, and B2C."""

    DATASET_DIR = Path("Datasets/rsicd")
    OUTPUT_DIR = Path("Datasets/enriched_caption")
    SPLITS = ("train", "valid", "test")

    # Pseudo-mask
    WORK_LONG_SIDE = 512
    USE_BILATERAL = True
    BILATERAL_D = 7
    BILATERAL_SIGMA_COLOR = 50
    BILATERAL_SIGMA_SPACE = 50
    MORPH_KERNEL = 3

    # M2B
    MIN_BOX_AREA = 64
    IGNORE_IDS: Set[int] = {0}

    # B2C
    CENTER_FRAC = 0.25
    N_CAPTIONS = 5
    MAX_SUBSET = 6
    COUNT_MANY_THRESHOLD = 10
    SEED = 0

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# COLOR VOCABULARY (kept from prior enricher)
# ============================================================================

RS_COLORS = {
    "green": {
        "hue_range": (60, 180),
        "description": "vegetation",
        "objects": ["trees", "grass", "forest", "park", "field"],
    },
    "dark_green": {
        "hue_range": (90, 150),
        "sat_min": 0.3,
        "val_max": 0.5,
        "description": "dense vegetation",
        "objects": ["forest", "dense trees"],
    },
    "light_green": {
        "hue_range": (60, 120),
        "sat_max": 0.5,
        "val_min": 0.5,
        "description": "grass or lawn",
        "objects": ["grass", "lawn", "field"],
    },
    "blue": {
        "hue_range": (180, 260),
        "description": "water",
        "objects": ["water", "river", "pond", "lake", "pool"],
    },
    "dark_blue": {
        "hue_range": (200, 250),
        "val_max": 0.4,
        "description": "deep water",
        "objects": ["lake", "river", "ocean"],
    },
    "gray": {
        "sat_max": 0.15,
        "description": "urban/roads",
        "objects": ["road", "pavement", "concrete", "building roof"],
    },
    "dark_gray": {
        "sat_max": 0.15,
        "val_max": 0.4,
        "description": "asphalt",
        "objects": ["road", "parking lot", "asphalt"],
    },
    "light_gray": {
        "sat_max": 0.15,
        "val_min": 0.6,
        "description": "concrete",
        "objects": ["concrete", "sidewalk", "building"],
    },
    "brown": {
        "hue_range": (10, 40),
        "description": "bare earth",
        "objects": ["bare soil", "dirt", "unpaved area", "farmland"],
    },
    "tan": {
        "hue_range": (30, 50),
        "val_min": 0.5,
        "description": "sand or dry ground",
        "objects": ["sand", "beach", "desert", "dry field"],
    },
    "red": {
        "hue_range": (0, 15),
        "description": "rooftops",
        "objects": ["roof", "building", "structure"],
    },
    "orange": {
        "hue_range": (15, 40),
        "sat_min": 0.4,
        "description": "clay roofs",
        "objects": ["clay roof", "terracotta", "building"],
    },
    "white": {
        "val_min": 0.85,
        "sat_max": 0.1,
        "description": "bright surfaces",
        "objects": ["building", "roof", "marking", "cloud shadow"],
    },
    "black": {
        "val_max": 0.15,
        "description": "shadows/dark areas",
        "objects": ["shadow", "dark structure"],
    },
}

COLOR_TO_GROUP = {
    "green": "vegetation",
    "dark_green": "vegetation",
    "light_green": "vegetation",
    "blue": "water",
    "dark_blue": "water",
    "gray": "urban",
    "dark_gray": "urban",
    "light_gray": "urban",
    "brown": "bare",
    "tan": "bare",
    "red": "built",
    "orange": "built",
    "white": "built",
    "black": "shadow",
    "mixed": "mixed",
}

# Class ids for the pseudo semantic mask (0 = background / mixed / ignore)
GROUP_TO_ID = {
    "mixed": 0,
    "vegetation": 1,
    "water": 2,
    "urban": 3,
    "bare": 4,
    "built": 5,
    "shadow": 6,
}

# Countable nouns for B2C phrasing
GROUP_TO_NOUN = {
    "vegetation": "patch of vegetation",
    "water": "body of water",
    "urban": "road",
    "bare": "bare ground area",
    "built": "building",
    "shadow": "shadowed area",
}

ID_TO_NOUN = {
    gid: GROUP_TO_NOUN[name]
    for name, gid in GROUP_TO_ID.items()
    if name != "mixed"
}

# Caption keyword grounding: when the original caption names a specific
# object type, remap dominant built/urban boxes to that noun.
CAPTION_SCENE_CUES = [
    (
        ["airport", "runway", "tarmac", "apron", "aircraft", "airplane", "plane", "planes"],
        "airport",
        "airplane",
    ),
    (["highway", "motorway", "freeway", "overpass", "interchange"], "highway", "road"),
    (["bridge", "viaduct"], "bridge", "bridge"),
    (["railway", "railroad", "train station", "rail station"], "railway", "railway"),
    (["parking lot", "car park", "parking"], "parking", "parking lot"),
    (["ocean", "sea", "coast", "beach", "shore"], "coast", "coastal water"),
    (["river", "stream", "canal"], "river", "river"),
    (["lake", "pond", "reservoir", "harbor", "harbour", "port", "dock"], "water", "body of water"),
    (["residential", "houses", "house", "apartment", "neighbourhood", "neighborhood"],
     "residential", "building"),
    (["industrial", "factory", "warehouse", "plant"], "industrial", "building"),
    (["commercial", "shopping", "market", "downtown"], "commercial", "building"),
    (["stadium", "arena", "sports field", "playground"], "stadium", "stadium"),
    (["school", "university", "campus"], "campus", "building"),
    (["forest", "woodland", "woods", "dense trees"], "forest", "patch of vegetation"),
    (["park", "lawn", "garden", "golf"], "park", "patch of vegetation"),
    (["farmland", "farm", "crop", "agricultural", "field", "fields", "meadow"],
     "agricultural", "field"),
    (["desert", "sand dune", "dune"], "desert", "bare ground area"),
    (["ship", "ships", "boat", "boats", "vessel"], "ship", "ship"),
    (["bare land", "bare ground", "bare soil", "naked land"], "bare", "bare ground area"),
]

# Which land-cover groups a caption cue may override
CUE_TARGET_GROUPS = {
    "airport": {"built", "urban", "bare"},
    "highway": {"urban", "built"},
    "bridge": {"urban", "built"},
    "railway": {"urban", "built"},
    "parking": {"urban", "built", "bare"},
    "coast": {"water"},
    "river": {"water"},
    "water": {"water"},
    "residential": {"built"},
    "industrial": {"built"},
    "commercial": {"built"},
    "stadium": {"built", "bare"},
    "campus": {"built"},
    "forest": {"vegetation"},
    "park": {"vegetation"},
    "agricultural": {"vegetation", "bare"},
    "desert": {"bare"},
    "ship": {"built", "water"},
    "bare": {"bare"},
}


def _rgb_array_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB->HSV. Input (N,3) 0-255; returns H 0-360, S/V 0-1."""
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
    maxc = arr.max(axis=1)
    minc = arr.min(axis=1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.where(maxc == 0, 1.0, maxc), 0.0)
    safe_delta = np.where(delta == 0, 1.0, delta)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    h = np.where(
        r == maxc,
        bc - gc,
        np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    h = (h / 6.0) % 1.0
    h = np.where(delta == 0, 0.0, h)
    return h * 360.0, s, v


def classify_colors_array(rgb: np.ndarray) -> np.ndarray:
    """
    Vectorized color classification matching RS_COLORS order (first match wins).
    Input rgb: (N, 3) uint8. Returns (N,) object array of color-name strings.
    """
    n = rgb.shape[0]
    h, s, v = _rgb_array_to_hsv(rgb)
    names = np.full(n, "mixed", dtype=object)
    assigned = np.zeros(n, dtype=bool)

    for color_name, props in RS_COLORS.items():
        match = np.ones(n, dtype=bool)
        if "hue_range" in props:
            h_min, h_max = props["hue_range"]
            match &= (h >= h_min) & (h <= h_max)
        if "sat_min" in props:
            match &= s >= props["sat_min"]
        if "sat_max" in props:
            match &= s <= props["sat_max"]
        if "val_min" in props:
            match &= v >= props["val_min"]
        if "val_max" in props:
            match &= v <= props["val_max"]
        take = match & ~assigned
        names[take] = color_name
        assigned |= take

    return names


def extract_caption_cues(text: str) -> Dict:
    """Keyword-only scene cues from an original caption (no ML)."""
    empty = {
        "scene_tag": None,
        "object_noun": None,
        "matched_keywords": [],
    }
    if not text or not text.strip():
        return empty

    lower = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    lower = re.sub(r"\s+", " ", lower).strip()

    for keywords, tag, noun in CAPTION_SCENE_CUES:
        hits = [kw for kw in keywords if kw in lower]
        if hits:
            return {
                "scene_tag": tag,
                "object_noun": noun,
                "matched_keywords": hits,
            }
    return empty


def normalize_label(name: str) -> str:
    """Normalize category names into readable text (large-vehicle -> large vehicle)."""
    return name.replace("-", " ").replace("_", " ").strip().lower()


# ============================================================================
# 1. PSEUDO MASK GENERATOR
# ============================================================================

class PseudoMaskGenerator:
    """
    Synthesize an approximate semantic segmentation mask from RGB
    using color land-cover classification + morphology (no ML).
    """

    def __init__(self, config: Config):
        self.config = config

    def generate(
        self, image: Image.Image
    ) -> Tuple[np.ndarray, Dict[int, str], float, float]:
        """
        Returns:
            mask: HxW uint8 class ids at working resolution
            class_id_to_name: id -> countable noun
            scale_x, scale_y: factors to map work coords -> original (orig = work * scale)
        """
        orig_w, orig_h = image.size
        work = image.convert("RGB")
        long_side = max(orig_w, orig_h)
        if long_side > self.config.WORK_LONG_SIDE:
            scale = self.config.WORK_LONG_SIDE / float(long_side)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            resample = getattr(Image, "Resampling", Image).BILINEAR
            work = work.resize((new_w, new_h), resample)
        else:
            new_w, new_h = orig_w, orig_h

        scale_x = orig_w / float(new_w)
        scale_y = orig_h / float(new_h)

        rgb = np.asarray(work, dtype=np.uint8)
        if self.config.USE_BILATERAL:
            rgb = cv2.bilateralFilter(
                rgb,
                d=self.config.BILATERAL_D,
                sigmaColor=self.config.BILATERAL_SIGMA_COLOR,
                sigmaSpace=self.config.BILATERAL_SIGMA_SPACE,
            )

        flat = rgb.reshape(-1, 3)
        color_names = classify_colors_array(flat)
        groups = np.array(
            [COLOR_TO_GROUP.get(n, "mixed") for n in color_names.tolist()],
            dtype=object,
        )
        mask = np.array(
            [GROUP_TO_ID.get(g, 0) for g in groups.tolist()],
            dtype=np.uint8,
        ).reshape(new_h, new_w)

        # Morphological cleanup per class (skip background)
        k = max(1, int(self.config.MORPH_KERNEL))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = np.zeros_like(mask)
        for gid in ID_TO_NOUN:
            binary = (mask == gid).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned[binary > 0] = gid

        return cleaned, dict(ID_TO_NOUN), scale_x, scale_y


# ============================================================================
# 2. M2B: MASK → BOXES
# ============================================================================

def mask_to_boxes(
    mask: np.ndarray,
    class_id_to_name: Dict[int, str],
    ignore_ids: Optional[Set[int]] = None,
    min_box_area: int = 64,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> List[Dict]:
    """
    Paper M2B: for each semantic class, extract external contours
    (Suzuki border-following via OpenCV), take axis-aligned bounding rect.

    mask: HxW int class ids (semantic). Each connected region of a class
    becomes one box; adjacent same-class objects merge (paper caveat).
    """
    ignore_ids = ignore_ids if ignore_ids is not None else {0}
    boxes: List[Dict] = []

    for class_id, class_name in class_id_to_name.items():
        if class_id in ignore_ids:
            continue
        binary = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        label = normalize_label(class_name)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < min_box_area:
                continue
            x1 = int(round(x * scale_x))
            y1 = int(round(y * scale_y))
            x2 = int(round((x + w) * scale_x))
            y2 = int(round((y + h) * scale_y))
            boxes.append({"label": label, "bbox_xyxy": [x1, y1, x2, y2]})

    return boxes


# ============================================================================
# 3. B2C: BOXES → CAPTIONS
# ============================================================================

def count_phrase(n: int, label: str, many_threshold: int = 10) -> str:
    """Exact count unless n > threshold -> broad phrase ('many')."""
    # Labels that already contain "patch of" / "body of" / "area" stay as-is
    # for pluralization; simple nouns get a trailing 's'.
    if n > many_threshold:
        return f"many {_pluralize(label)}"
    if n == 1:
        return f"one {label}"
    return f"{n} {_pluralize(label)}"


_IRREGULAR_PLURALS = {
    "body of water": "bodies of water",
    "patch of vegetation": "patches of vegetation",
    "bare ground area": "bare ground areas",
    "shadowed area": "shadowed areas",
    "parking lot": "parking lots",
}


def _pluralize(label: str) -> str:
    if label in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[label]
    if label.endswith("s"):
        return label
    if label.endswith(" area"):
        return label + "s"
    if " of " in label:
        # "X of Y" -> pluralize the head noun
        head, rest = label.split(" of ", 1)
        return f"{_pluralize(head)} of {rest}"
    if label.endswith("y") and not label.endswith(("ay", "ey", "oy", "uy")):
        return label[:-1] + "ies"
    return label + "s"


def summarize(
    items: Sequence[str], many_threshold: int = 10
) -> str:
    counts = Counter(items)
    return ", ".join(
        count_phrase(n, label, many_threshold) for label, n in counts.items()
    )


def boxes_to_captions(
    boxes: List[Dict],
    image_w: int,
    image_h: int,
    n_captions: int = 5,
    center_frac: float = 0.25,
    max_subset: int = 6,
    many_threshold: int = 10,
    seed: Optional[int] = 0,
) -> List[str]:
    """
    RemoteCLIP B2C: up to 2 spatial captions (center / away) plus random
    object-subset captions, totaling exactly n_captions strings.
    """
    rng = random.Random(seed)
    cx0, cy0 = image_w / 2.0, image_h / 2.0

    central: List[str] = []
    peripheral: List[str] = []
    all_labs: List[str] = []

    for box in boxes:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        lab = normalize_label(box["label"])
        all_labs.append(lab)
        if abs(cx - cx0) < center_frac * image_w and abs(cy - cy0) < center_frac * image_h:
            central.append(lab)
        else:
            peripheral.append(lab)

    captions: List[str] = []
    if central:
        captions.append(
            f"In the center of the image, there are {summarize(central, many_threshold)}."
        )
    if peripheral:
        captions.append(
            f"Away from the center, there are {summarize(peripheral, many_threshold)}."
        )

    while len(captions) < n_captions and all_labs:
        k = rng.randint(1, min(len(all_labs), max_subset))
        sample = rng.sample(all_labs, k)
        captions.append(
            f"The image contains {summarize(sample, many_threshold)}."
        )

    while len(captions) < n_captions:
        if all_labs:
            captions.append(
                f"The image contains {summarize(all_labs, many_threshold)}."
            )
        else:
            captions.append("An aerial remote sensing image.")

    return captions[:n_captions]


# ============================================================================
# LABEL GROUNDING FROM CAPTION KEYWORDS
# ============================================================================

def ground_box_labels(
    boxes: List[Dict],
    caption: str,
    mask: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Optionally remap box labels using caption keywords.
    e.g. caption mentions airplanes -> built/urban boxes near airports
    become 'airplane' when the cue targets those groups.
    """
    cues = extract_caption_cues(caption)
    tag = cues.get("scene_tag")
    noun = cues.get("object_noun")
    if not tag or not noun:
        return boxes

    targets = CUE_TARGET_GROUPS.get(tag)
    if not targets:
        return boxes

    # Invert GROUP_TO_NOUN for matching current labels
    noun_to_group = {normalize_label(v): k for k, v in GROUP_TO_NOUN.items()}
    grounded = []
    for box in boxes:
        lab = normalize_label(box["label"])
        group = noun_to_group.get(lab)
        if group in targets:
            grounded.append({"label": normalize_label(noun), "bbox_xyxy": box["bbox_xyxy"]})
        else:
            grounded.append(box)
    return grounded


# ============================================================================
# PIPELINE ENGINE
# ============================================================================

class M2BB2CPipeline:
    """End-to-end: RGB (+ optional caption) -> 5 B2C captions + boxes."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.masker = PseudoMaskGenerator(self.config)

    def process_image(
        self,
        image_path: str,
        original_caption: str = "",
        seed: Optional[int] = None,
    ) -> Dict:
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        mask, class_map, scale_x, scale_y = self.masker.generate(image)
        boxes = mask_to_boxes(
            mask,
            class_map,
            ignore_ids=self.config.IGNORE_IDS,
            min_box_area=self.config.MIN_BOX_AREA,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        if original_caption:
            boxes = ground_box_labels(boxes, original_caption)

        use_seed = self.config.SEED if seed is None else seed
        captions = boxes_to_captions(
            boxes,
            img_w,
            img_h,
            n_captions=self.config.N_CAPTIONS,
            center_frac=self.config.CENTER_FRAC,
            max_subset=self.config.MAX_SUBSET,
            many_threshold=self.config.COUNT_MANY_THRESHOLD,
            seed=use_seed,
        )

        return {
            "image_path": str(image_path),
            "original_caption": original_caption,
            "captions": captions,
            "boxes": boxes,
            "box_count": len(boxes),
            "caption_cues": extract_caption_cues(original_caption),
        }

    def process_split(
        self,
        split: str,
        dataset_dir: Path,
        output_dir: Path,
        max_images: Optional[int] = None,
        rich: bool = False,
    ) -> Path:
        """
        Read <dataset_dir>/<split>.jsonl, write OpenCLIP-style pairs to
        <output_dir>/<split>.jsonl (5 lines per image). Crash-safe incremental write.
        If rich=True, also write <output_dir>/<split>_rich.jsonl with boxes.
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        src_jsonl = dataset_dir / f"{split}.jsonl"
        out_jsonl = output_dir / f"{split}.jsonl"
        rich_jsonl = output_dir / f"{split}_rich.jsonl"

        if not src_jsonl.exists():
            print(f"  ✗ Skipping '{split}': {src_jsonl} not found")
            return out_jsonl

        entries = []
        with open(src_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        if max_images is not None:
            entries = entries[:max_images]

        print(f"\n{'=' * 60}")
        print(f"Split: {split}")
        print(f"{'=' * 60}")
        print(f"Source jsonl : {src_jsonl}")
        print(f"Output jsonl : {out_jsonl}")
        print(f"Entries      : {len(entries)}")
        print(f"{'=' * 60}\n")

        ok, failed = 0, 0
        rich_f = open(rich_jsonl, "w", encoding="utf-8") if rich else None
        try:
            with open(out_jsonl, "w", encoding="utf-8") as out_f:
                for idx, entry in enumerate(entries):
                    rel_image = entry.get("image", "")
                    img_path = dataset_dir / rel_image
                    orig_caption = entry.get("caption", "")
                    # Stable per-image seed from id or index
                    seed = entry.get("id", idx)

                    try:
                        result = self.process_image(
                            str(img_path), orig_caption, seed=seed
                        )
                        for cap in result["captions"]:
                            out_f.write(
                                json.dumps(
                                    {"image": rel_image, "caption": cap},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        if rich_f is not None:
                            rich_f.write(
                                json.dumps(
                                    {
                                        "id": entry.get("id", idx),
                                        "image": rel_image,
                                        "original_caption": orig_caption,
                                        "captions": result["captions"],
                                        "boxes": result["boxes"],
                                        "box_count": result["box_count"],
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        ok += 1
                        if (idx + 1) % 50 == 0 or idx == 0:
                            sample = result["captions"][0] if result["captions"] else ""
                            print(
                                f"  [{idx + 1}/{len(entries)}] {rel_image} "
                                f"({result['box_count']} boxes) -> {sample[:70]}..."
                            )
                    except Exception as e:
                        failed += 1
                        print(f"  ✗ [{idx + 1}/{len(entries)}] {rel_image}: {e}")
                        # Fall back: keep original caption once so training isn't empty
                        fallback = orig_caption or "An aerial remote sensing image."
                        out_f.write(
                            json.dumps(
                                {"image": rel_image, "caption": fallback},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        finally:
            if rich_f is not None:
                rich_f.close()

        print(f"\n  ✓ {split}: {ok} ok, {failed} failed -> {out_jsonl}")
        if rich:
            print(f"  ✓ Rich records -> {rich_jsonl}")
        return out_jsonl

    def process_rsicd(
        self,
        dataset_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        splits: Optional[Tuple[str, ...]] = None,
        max_images: Optional[int] = None,
        rich: bool = False,
    ) -> None:
        dataset_dir = Path(dataset_dir or self.config.DATASET_DIR)
        output_dir = Path(output_dir or self.config.OUTPUT_DIR)
        splits = splits or self.config.SPLITS

        print(f"\n{'#' * 60}")
        print("RemoteCLIP M2B + B2C Caption Pipeline")
        print(f"{'#' * 60}")
        print(f"Dataset : {dataset_dir}")
        print(f"Output  : {output_dir}")
        print(f"Splits  : {', '.join(splits)}")

        for split in splits:
            self.process_split(
                split,
                dataset_dir,
                output_dir,
                max_images=max_images,
                rich=rich,
            )

        print(f"\n{'#' * 60}")
        print(f"✓ Done. OpenCLIP pairs in: {output_dir}")
        print(f"{'#' * 60}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_boxes(
    image_path: str,
    boxes: List[Dict],
    output_path: Optional[str] = None,
) -> Image.Image:
    """Draw M2B boxes + labels on the image for debugging."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = [
        "red", "blue", "green", "yellow", "purple",
        "orange", "cyan", "magenta", "lime", "pink",
    ]
    for idx, box in enumerate(boxes):
        color = colors[idx % len(colors)]
        x1, y1, x2, y2 = box["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), box.get("label", str(idx)), fill=color)
    if output_path:
        image.save(output_path)
        print(f"Visualization saved to: {output_path}")
    return image


# ============================================================================
# CLI
# ============================================================================

def main():
    config = Config()
    parser = argparse.ArgumentParser(
        description="RemoteCLIP-style M2B + B2C caption pipeline"
    )
    parser.add_argument(
        "--dataset", "-d",
        default=str(config.DATASET_DIR),
        help="RSICD dataset root (contains <split>.jsonl and <split>/images)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=str(config.OUTPUT_DIR),
        help="Directory for OpenCLIP-style <split>.jsonl files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(config.SPLITS),
        help="Which splits to process (default: train valid test)",
    )
    parser.add_argument(
        "--single", "-s",
        default=None,
        help="Process a single image path instead of the dataset",
    )
    parser.add_argument(
        "--caption",
        default="",
        help="Original caption for --single (optional, for label grounding)",
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Draw boxes on the image (single mode only)",
    )
    parser.add_argument(
        "--max-images", "-m",
        type=int,
        default=None,
        help="Maximum number of images to process per split",
    )
    parser.add_argument(
        "--rich",
        action="store_true",
        help="Also write <split>_rich.jsonl with boxes and all captions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="RNG seed for B2C random subset captions",
    )

    args = parser.parse_args()
    config.SEED = args.seed
    pipeline = M2BB2CPipeline(config)

    if args.single:
        result = pipeline.process_image(args.single, args.caption, seed=args.seed)
        print(f"\nOriginal: {result['original_caption']}")
        print(f"Boxes   : {result['box_count']}")
        for i, cap in enumerate(result["captions"], 1):
            print(f"  [{i}] {cap}")
        if args.visualize:
            stem = Path(args.single)
            vis_path = str(stem.with_name(stem.stem + "_boxes" + stem.suffix))
            visualize_boxes(args.single, result["boxes"], vis_path)
    else:
        pipeline.process_rsicd(
            dataset_dir=args.dataset,
            output_dir=args.output_dir,
            splits=tuple(args.splits),
            max_images=args.max_images,
            rich=args.rich,
        )


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
