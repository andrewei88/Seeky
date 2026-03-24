#!/usr/bin/env python3
"""Supplement training data with web images for classes that perform poorly.

Uses icrawler to download images from Bing for specific search queries
designed to match what a phone camera sees in a home environment.

Usage:
    python scripts/supplement_training_data.py
"""

import os
import random
import shutil
from pathlib import Path

from icrawler.builtin import BingImageCrawler
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "arya_training"
CROP_SIZE = (256, 256)
VAL_FRACTION = 0.15
IMAGES_PER_QUERY = 80  # Bing returns ~100 max per query

# Each class gets multiple search queries to capture variety.
# Queries are designed to match what a phone camera sees in a home.
SUPPLEMENTS = {
    "panda": [
        "stuffed panda toy",
        "panda plush toy",
        "panda stuffed animal",
        "panda teddy bear toy",
        "cute panda plushie",
        "black white panda toy",
    ],
    "light": [
        "ceiling light fixture",
        "ceiling light home",
        "room ceiling light",
        "LED ceiling light",
        "recessed ceiling light",
        "pendant light hanging",
        "ceiling light looking up",
        "light fixture on ceiling photo",
    ],
    "monitor": [
        "computer monitor on desk",
        "desktop monitor screen",
        "PC monitor front view",
        "computer screen display",
        "office monitor close up",
        "flat screen computer monitor",
        "monitor showing website",
        "monitor with text on screen",
        "computer monitor angle view desk",
        "iMac on desk",
    ],
    "laptop": [
        "open laptop on table",
        "laptop computer screen",
        "laptop keyboard and screen",
        "MacBook on desk",
        "laptop front view",
        "open laptop close up",
        "laptop showing website screen",
        "laptop with text on screen",
        "laptop on desk from above phone photo",
        "MacBook Pro open on table angle",
        "laptop screen glowing in room",
        "open laptop side angle desk",
    ],
    "lamp": [
        "table lamp bedroom",
        "desk lamp close up",
        "bedside lamp",
        "floor lamp living room",
        "reading lamp",
        "lamp shade close up",
    ],
    "speaker": [
        "bluetooth speaker on table",
        "portable speaker close up",
        "wireless speaker on desk",
        "smart speaker home",
        "JBL speaker",
        "bookshelf speaker",
        "small speaker on table",
        "speaker front view",
    ],
}


def download_images(query: str, output_dir: Path, max_num: int = IMAGES_PER_QUERY):
    """Download images from Bing for a search query."""
    output_dir.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(
        storage={"root_dir": str(output_dir)},
        log_level="WARNING",
    )
    crawler.crawl(keyword=query, max_num=max_num)


def split_class(word: str):
    """Re-split a single class into train/val."""
    random.seed(42)
    word_slug = word.replace(" ", "_")
    all_dir = OUTPUT_DIR / "all" / word_slug
    train_dir = OUTPUT_DIR / "train" / word_slug
    val_dir = OUTPUT_DIR / "val" / word_slug

    if not all_dir.exists():
        print(f"  [WARN] No all/ directory for '{word}'")
        return

    # Clear old split
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)

    images = sorted(all_dir.glob("*.jpg"))
    random.shuffle(images)

    val_count = max(1, int(len(images) * VAL_FRACTION))
    val_images = images[:val_count]
    train_images = images[val_count:]

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for p in train_images:
        shutil.copy2(p, train_dir / p.name)
    for p in val_images:
        shutil.copy2(p, val_dir / p.name)

    print(f"  {word}: {len(train_images)} train, {len(val_images)} val (total: {len(images)})")


def main():
    print("Supplementing training data with web images")
    print(f"Output: {OUTPUT_DIR}")
    print()

    for word, queries in SUPPLEMENTS.items():
        print(f"\n{'='*60}")
        print(f"Supplementing: {word} ({len(queries)} queries)")
        print(f"{'='*60}")

        word_slug = word.replace(" ", "_")
        all_dir = OUTPUT_DIR / "all" / word_slug
        existing = len(list(all_dir.glob("*.jpg"))) if all_dir.exists() else 0
        print(f"  Existing images: {existing}")

        # Download to a temp dir, then process
        tmp_dir = OUTPUT_DIR / "tmp_download"

        total_new = 0
        for query in queries:
            print(f"  Downloading: '{query}'...")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            download_images(query, tmp_dir)

            # Resize and move to all/ dir
            all_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for img_path in sorted(tmp_dir.glob("*")):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(CROP_SIZE, Image.LANCZOS)
                    new_name = f"{word}_web_{existing + total_new + count:05d}.jpg"
                    img.save(all_dir / new_name, "JPEG", quality=90)
                    count += 1
                except Exception:
                    pass
            total_new += count
            print(f"    Got {count} images")

        # Cleanup
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        print(f"  Added {total_new} supplemental images (total: {existing + total_new})")

        # Re-split
        split_class(word)

    print(f"\n{'='*60}")
    print("SUPPLEMENTATION COMPLETE")
    print(f"{'='*60}")
    print("\nNext: python scripts/train_classifier.py")


if __name__ == "__main__":
    main()
