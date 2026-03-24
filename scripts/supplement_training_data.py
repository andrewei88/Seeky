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
        # Round 3: non-plush pandas — wall art, rugs, prints
        "panda wall hanging tapestry",
        "panda rug nursery",
        "panda print on blanket",
        "panda face pillow",
        "panda wall decor kids room",
        "panda embroidered patch close up",
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
        # Round 2: improve recognition from desk angle
        "ceiling light from below bright",
        "overhead light looking up from desk",
        "flush mount ceiling light",
        "round ceiling light fixture white",
        "kitchen ceiling light bright",
        "bathroom ceiling light close",
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
        # Round 2: screen content visible — teach model to see the device frame
        "monitor with code on screen",
        "monitor displaying spreadsheet",
        "monitor with dark mode website",
        "dual monitor setup showing apps",
        "monitor with colorful wallpaper",
        "Dell monitor front view on desk",
        "monitor with email open",
        "computer monitor with video playing",
        "monitor screen with browser tabs",
        "wide monitor with multiple windows",
        # Round 3: close-up partial views, bezels visible, oblique angles
        "monitor bezel close up",
        "monitor edge side view desk",
        "computer screen close up text",
        "monitor from below looking up",
        "monitor at angle showing reflection",
        "iMac screen close up with dock",
        "monitor partial view with keyboard",
        "desktop monitor zoomed in corner",
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
        # Round 2: screen content visible — teach model to see the physical laptop
        "laptop with code editor open",
        "laptop showing YouTube video",
        "laptop with dark screen in room",
        "MacBook with browser open angle",
        "laptop displaying presentation slides",
        "laptop with Zoom meeting on screen",
        "laptop screen with bright colors",
        "thin laptop open on wooden desk",
        "laptop from slight angle with website",
        "laptop open next to coffee cup",
        # Round 3: oblique/partial views, close-ups showing screen + frame edge
        "laptop side angle showing screen edge",
        "MacBook half closed angle view",
        "laptop from the side on desk",
        "laptop screen and hinge close up",
        "laptop oblique angle with text on screen",
        "laptop close up screen with taskbar",
        "laptop partial view from right side",
        "laptop screen edge and keyboard corner",
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
    "doll": [
        # Round 3: small figurines and plush dolls — prevent confusion with panda/bear
        "small doll figurine close up",
        "stuffed animal doll on shelf",
        "cute plush doll toy",
        "action figure toy close up",
        "small doll on table",
        "plush character toy on bed",
        "miniature doll in costume",
        "stuffed doll toy for kids",
    ],
    "paper": [
        # Round 3: actual paper/documents — NOT screens
        "sheet of paper on desk",
        "white paper on table",
        "paper document on wooden desk",
        "notebook paper close up",
        "piece of paper with writing",
        "blank paper on desk from above",
        "paper with text on table",
        "crumpled paper on desk",
    ],
    "glass": [
        # Round 2: transparent/reflective objects are hard to classify
        "drinking glass on table",
        "empty glass on counter",
        "water glass close up",
        "clear glass on wooden table",
        "glass of water on desk",
        "drinking glass kitchen counter",
        "tall glass tumbler",
        "short glass on table",
        "glass cup transparent",
        "clear drinking glass side view",
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
