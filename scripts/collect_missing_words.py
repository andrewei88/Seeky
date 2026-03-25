#!/usr/bin/env python3
"""Collect images for the 18 words the main script missed.

Strategy:
1. Try corrected OI7 labels for words where label name was wrong
2. For remaining words, download from web using bing_image_downloader
3. Resize and crop all downloads to 256x256

Usage:
    pip install bing-image-downloader
    python scripts/collect_missing_words.py
"""

import os
import shutil
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "arya_training" / "all"
TARGET = 500
CROP_SIZE = (256, 256)
MIN_PX = 100  # minimum image dimension to keep

# Corrected OI7 labels (discovered via get_classes)
OI7_FIXABLE = {
    "egg": [("oi7", "Egg (Food)")],
}

# Words to download from web search with search queries
WEB_DOWNLOAD = {
    "blanket":  ["blanket on bed", "folded blanket", "baby blanket", "blanket cozy"],
    "block":    ["toy building blocks", "wooden blocks children", "colorful toy blocks", "stacking blocks toddler"],
    "cloud":    ["cloud in sky", "white cloud blue sky", "single cloud", "fluffy cloud"],
    "crayon":   ["crayons", "crayon box", "coloring with crayons", "wax crayons"],
    "fence":    ["wooden fence", "garden fence", "white picket fence", "fence outdoor"],
    "grass":    ["green grass lawn", "grass field", "grass close up", "grass texture"],
    "key":      ["door key", "metal key", "house key", "key on table"],
    "leaf":     ["green leaf", "tree leaf", "autumn leaf", "leaf close up"],
    "moon":     ["full moon", "moon in sky", "crescent moon", "moon night"],
    "paper":    ["sheet of paper", "white paper", "paper on desk", "paper stack"],
    "pencil":   ["pencil", "writing pencil", "pencils on desk", "yellow pencil"],
    "rain":     ["rain drops", "rain on window", "raining", "rain puddle"],
    "rock":     ["rock stone", "rock on ground", "pebble stone", "garden rock"],
    "soap":     ["bar of soap", "soap bar", "hand soap", "soap bathroom"],
    "speaker":  ["bluetooth speaker", "speaker audio", "portable speaker", "wireless speaker"],
    "star":     ["star in sky", "night star", "star shape", "twinkling star"],
    "sun":      ["sun in sky", "sunrise", "bright sun", "sun rays"],
}


def collect_from_oi7_fixed():
    """Collect images using corrected OI7 labels."""
    import fiftyone.zoo as foz

    for word, sources in OI7_FIXABLE.items():
        word_dir = OUTPUT_DIR / word
        word_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(word_dir.glob("*.jpg")))

        if existing >= TARGET:
            print(f"[{word}] Already have {existing}, skipping")
            continue

        total = existing
        for ds_name, label in sources:
            if total >= TARGET:
                break
            print(f"[{word}] Trying OI7 '{label}'...")
            sys.stdout.flush()

            try:
                dataset = foz.load_zoo_dataset(
                    "open-images-v7",
                    split="train",
                    label_types=["detections"],
                    classes=[label],
                    max_samples=min(2000, (TARGET - total) * 4),
                    shuffle=True,
                    dataset_name=None,
                )

                saved = 0
                for sample in dataset:
                    if total + saved >= TARGET:
                        break
                    if not os.path.exists(sample.filepath):
                        continue
                    dets = sample.ground_truth.detections if sample.ground_truth else []
                    matching = [d for d in dets if d.label == label]
                    if not matching:
                        continue

                    try:
                        img = Image.open(sample.filepath).convert("RGB")
                    except Exception:
                        continue

                    w, h = img.size
                    for det in matching:
                        if total + saved >= TARGET:
                            break
                        bx, by, bw, bh = det.bounding_box
                        x1, y1 = int(bx * w), int(by * h)
                        x2, y2 = int((bx + bw) * w), int((by + bh) * h)
                        if (x2 - x1) < 20 or (y2 - y1) < 20:
                            continue
                        pad_x = int((x2 - x1) * 0.1)
                        pad_y = int((y2 - y1) * 0.1)
                        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
                        crop = img.crop((x1, y1, x2, y2)).resize(CROP_SIZE, Image.LANCZOS)
                        fname = f"{word}_{total + saved:05d}.jpg"
                        crop.save(word_dir / fname, "JPEG", quality=90)
                        saved += 1

                total += saved
                print(f"  Got {saved} crops (total: {total})")
                dataset.delete()
            except Exception as e:
                print(f"  Failed: {e}")

        print(f"[{word}] Final: {total}")
        sys.stdout.flush()


def collect_from_web():
    """Download images from Bing Image Search."""
    try:
        from bing_image_downloader import downloader
    except ImportError:
        print("Install: pip install bing-image-downloader")
        return

    tmp_dir = PROJECT_ROOT / "data" / "tmp_downloads"

    for word, queries in WEB_DOWNLOAD.items():
        word_dir = OUTPUT_DIR / word
        word_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(word_dir.glob("*.jpg")))

        if existing >= TARGET:
            print(f"[{word}] Already have {existing}, skipping")
            continue

        print(f"\n[{word}] Downloading from web ({existing} existing)...")
        sys.stdout.flush()

        total = existing
        for query in queries:
            if total >= TARGET:
                break

            needed = TARGET - total
            per_query = min(200, needed)

            try:
                downloader.download(
                    query,
                    limit=per_query,
                    output_dir=str(tmp_dir),
                    adult_filter_off=False,
                    force_replace=False,
                    timeout=10,
                )
            except Exception as e:
                print(f"  Query '{query}' failed: {e}")
                continue

            # Process downloaded images
            query_dir = tmp_dir / query
            if not query_dir.exists():
                continue

            for img_path in sorted(query_dir.iterdir()):
                if total >= TARGET:
                    break
                try:
                    img = Image.open(img_path).convert("RGB")
                    w, h = img.size
                    if w < MIN_PX or h < MIN_PX:
                        continue

                    # Center crop to square, then resize
                    side = min(w, h)
                    left = (w - side) // 2
                    top = (h - side) // 2
                    img = img.crop((left, top, left + side, top + side))
                    img = img.resize(CROP_SIZE, Image.LANCZOS)

                    fname = f"{word}_{total:05d}.jpg"
                    img.save(word_dir / fname, "JPEG", quality=90)
                    total += 1
                except Exception:
                    continue

            print(f"  '{query}': now at {total}")
            sys.stdout.flush()

        print(f"[{word}] Final: {total}")

    # Clean up temp downloads
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def main():
    print("=== Phase 1: Corrected OI7 labels ===")
    collect_from_oi7_fixed()

    print("\n=== Phase 2: Web image downloads ===")
    collect_from_web()

    # Final report
    print("\n=== FINAL REPORT ===")
    all_missing = list(OI7_FIXABLE.keys()) + list(WEB_DOWNLOAD.keys())
    for word in sorted(set(all_missing)):
        word_dir = OUTPUT_DIR / word
        count = len(list(word_dir.glob("*.jpg"))) if word_dir.exists() else 0
        status = "OK" if count >= 200 else "LOW"
        print(f"  {word}: {count} [{status}]")

    print("\nAfter collecting, re-run to split:")
    print("  python scripts/collect_training_data.py  # will re-split train/val")


if __name__ == "__main__":
    main()
