#!/usr/bin/env python3
"""Collect additional TV training images to balance the tv/monitor split.

TV only has ~481 images vs monitor's ~5917. This script pulls more from
Open Images V7 and COCO by requesting larger sample sets.

Usage:
    python scripts/collect_tv_data.py
"""

import os
import sys
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "arya_training" / "all" / "tv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CROP_SIZE = (256, 256)
MIN_CROP_PX = 20
TARGET_TOTAL = 2000  # Want ~2000 images total for TV

# Sources: request many more samples than before
SOURCES = [
    ("oi7", "Television", 5000),
    ("oi7", "Plasma tv", 2000),
    ("coco", "tv", 3000),
]


def crop_and_save(dataset, label, output_dir, existing_count, max_total):
    """Crop bounding boxes from dataset, save as images. Returns count saved."""
    saved = 0
    for sample in dataset:
        if existing_count + saved >= max_total:
            break
        if sample.filepath is None or not os.path.exists(sample.filepath):
            continue
        detections = sample.ground_truth.detections if sample.ground_truth else []
        matching = [d for d in detections if d.label == label]
        if not matching:
            continue
        try:
            img = Image.open(sample.filepath).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        for det in matching:
            if existing_count + saved >= max_total:
                break
            bx, by, bw, bh = det.bounding_box
            x1 = int(bx * w)
            y1 = int(by * h)
            x2 = int((bx + bw) * w)
            y2 = int((by + bh) * h)
            if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                continue
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            crop = img.crop((x1, y1, x2, y2))
            crop = crop.resize(CROP_SIZE, Image.LANCZOS)
            fname = f"tv_{existing_count + saved:05d}.jpg"
            crop.save(output_dir / fname, "JPEG", quality=90)
            saved += 1
    return saved


def main():
    existing = len(list(OUTPUT_DIR.glob("*.jpg")))
    print(f"Existing TV images: {existing}")

    if existing >= TARGET_TOTAL:
        print(f"Already have {existing} >= {TARGET_TOTAL}, done")
        return

    total = existing

    for dataset_name, label, max_samples in SOURCES:
        if total >= TARGET_TOTAL:
            break

        print(f"\nFetching '{label}' from {dataset_name} (up to {max_samples} samples)...")

        try:
            if dataset_name == "oi7":
                dataset = foz.load_zoo_dataset(
                    "open-images-v7",
                    split="train",
                    label_types=["detections"],
                    classes=[label],
                    max_samples=max_samples,
                    shuffle=True,
                    dataset_name=None,
                )
            elif dataset_name == "coco":
                dataset = foz.load_zoo_dataset(
                    "coco-2017",
                    split="train",
                    label_types=["detections"],
                    classes=[label],
                    max_samples=max_samples,
                    shuffle=True,
                    dataset_name=None,
                )
            else:
                continue
        except Exception as e:
            print(f"  [WARN] Failed to load {dataset_name} '{label}': {e}")
            continue

        saved = crop_and_save(dataset, label, OUTPUT_DIR, total, TARGET_TOTAL)
        total += saved
        print(f"  Got {saved} crops (total: {total})")
        sys.stdout.flush()

        dataset.delete()

    print(f"\nFinal count: {total} TV images")
    if total < TARGET_TOTAL:
        print(f"  WARNING: Only got {total}/{TARGET_TOTAL}. OI7/COCO may not have enough TV images.")
        print(f"  Consider supplementing with web-scraped or phone-captured TV images.")


if __name__ == "__main__":
    main()
