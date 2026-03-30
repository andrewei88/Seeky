#!/usr/bin/env python3
"""Collect close-up, object-centric training images for confused classes.

The home-context collection (collect_home_context_data.py) used room-level queries
that returned images where the target object was small in the frame. This caused
regressions for cup, bowl, glass, book, and table — the model couldn't learn clear
boundaries between visually similar objects.

This script uses close-up queries where the object fills most of the frame.

Usage:
    python scripts/collect_closeup_data.py
    python scripts/collect_closeup_data.py --word cup --max-per-query 40
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "seeky_training" / "all"
CROP_SIZE = (256, 256)

# Close-up, object-centric queries.
# Each query is designed so the object fills most of the resulting image.
# Avoids room-level scenes where the object is a small part of the frame.
CLOSEUP_QUERIES = {
    "ball": [
        "ball close up",
        "soccer ball close up",
        "basketball close up",
        "tennis ball close up",
        "rubber ball close up",
        "colorful ball close up",
        "beach ball close up",
        "toy ball close up",
        "ball on grass close up",
        "ball on floor close up",
        "red ball close up",
        "ball held in hand",
        "bouncy ball close up",
        "kids ball close up",
        "ball on table close up",
    ],
    "cup": [
        "coffee mug close up",
        "coffee cup close up on table",
        "cup with handle close up",
        "tea cup close up",
        "sippy cup close up",
        "ceramic mug close up",
        "cup held in hand",
        "colorful mug close up",
        "travel mug close up",
        "cup on saucer close up",
        "coffee mug top view",
        "kids cup close up",
    ],
    "glass": [
        "drinking glass close up",
        "glass of water close up",
        "empty glass on table close up",
        "glass tumbler close up",
        "clear glass close up",
        "glass with ice close up",
        "tall glass close up",
        "juice glass close up",
        "glass from above close up",
        "wine glass close up",
    ],
    "bowl": [
        "bowl close up from above",
        "cereal bowl close up",
        "empty bowl close up",
        "soup bowl close up",
        "white bowl close up",
        "bowl with spoon close up",
        "mixing bowl close up",
        "salad bowl close up",
        "rice bowl close up",
        "bowl on table close up",
    ],
    "book": [
        "book cover close up",
        "book on table close up",
        "closed book close up",
        "book spine close up",
        "picture book close up",
        "children book close up",
        "stack of books close up",
        "book pages close up",
        "book standing up close up",
        "hardcover book close up",
    ],
    "table": [
        "table surface close up wood",
        "wooden table top close up",
        "table edge close up",
        "kitchen table close up",
        "table corner close up",
        "dining table surface close up",
        "table from above close up",
        "round table close up",
        "table with nothing on it",
        "clean table surface close up",
    ],
    "basketball": [
        "basketball close up",
        "basketball on court",
        "basketball on ground",
        "basketball held in hands",
        "basketball on grass",
        "orange basketball",
        "basketball texture seams",
        "basketball on floor",
        "basketball outdoor",
        "basketball in gym",
        "kids basketball",
        "mini basketball",
        "basketball on table",
        "basketball indoors",
        "basketball on wood floor",
    ],
    "soccer_ball": [
        "soccer ball close up",
        "soccer ball on grass",
        "soccer ball on field",
        "soccer ball black white",
        "soccer ball on ground",
        "soccer ball held in hands",
        "soccer ball outdoor",
        "soccer ball close up texture",
        "kids soccer ball",
        "soccer ball on floor",
        "soccer ball on dirt",
        "soccer ball in yard",
        "football soccer ball",
        "soccer ball indoor",
        "mini soccer ball",
    ],
    "tennis_ball": [
        "tennis ball close up",
        "tennis ball on court",
        "tennis ball on grass",
        "tennis ball on ground",
        "tennis ball green yellow",
        "tennis ball close up texture",
        "tennis ball held in hand",
        "tennis ball on table",
        "tennis ball fuzzy",
        "dog tennis ball",
        "tennis ball on floor",
        "tennis ball outdoor",
        "tennis ball indoors",
        "tennis ball can",
        "tennis ball on carpet",
    ],
    "cup_distance": [
        "cups on kitchen counter from far away",
        "coffee mugs on shelf distance",
        "cups on table from across room",
        "mugs on counter far shot",
        "cups in kitchen wide shot",
        "cup on desk from doorway",
        "cups on dining table wide angle",
        "coffee cups in cafe wide shot",
        "mug on counter kitchen scene",
        "cups on shelf from distance",
    ],
    "panda_toy": [
        "stuffed panda toy",
        "panda plush toy close up",
        "panda stuffed animal",
        "panda teddy bear plush",
        "stuffed panda on couch",
        "panda plush on bed",
        "stuffed panda on floor",
        "panda toy on shelf",
        "black white panda plush",
        "panda stuffed animal close up",
        "cute panda plush toy",
        "panda bear stuffed animal",
        "plush panda toy sitting",
        "stuffed panda toy face",
        "panda toy on blanket",
        "panda plush in room",
        "baby panda stuffed animal",
        "round panda plush",
        "panda toy body close up",
        "stuffed panda from above",
    ],
}


def download_bing_images(query: str, max_images: int = 50) -> list[Path]:
    """Download images from Bing Image Search using icrawler."""
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("  pip install icrawler  (required for Bing image search)")
        sys.exit(1)

    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="seeky_closeup_"))

    crawler = BingImageCrawler(
        storage={"root_dir": str(tmp_dir)},
        log_level="WARNING",
    )
    crawler.crawl(keyword=query, max_num=max_images)

    return list(tmp_dir.glob("*"))


def process_and_save(image_paths: list[Path], word: str, word_dir: Path) -> int:
    """Process downloaded images and save as training data."""
    saved = 0

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        w, h = img.size
        if w < 50 or h < 50:
            continue

        # Resize to training size
        img = img.resize(CROP_SIZE, Image.LANCZOS)

        # Use content hash to avoid duplicates
        content_hash = hashlib.md5(img.tobytes()[:10000]).hexdigest()[:8]
        fname = f"{word}_closeup_{content_hash}.jpg"
        out_path = word_dir / fname

        if out_path.exists():
            continue

        img.save(out_path, "JPEG", quality=90)
        saved += 1

    return saved


def collect_for_word(word: str, queries: list[str], max_per_query: int = 40):
    """Collect close-up images for a word."""
    # Some query sets map to a different training directory
    WORD_TO_DIR = {"cup_distance": "cup", "panda_toy": "panda"}
    target_word = WORD_TO_DIR.get(word, word)
    word_dir = OUTPUT_DIR / target_word.replace(" ", "_")
    word_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(word_dir.glob("*.jpg")))
    print(f"\n[{word}] Existing: {existing} images")

    total_new = 0
    for query in queries:
        print(f"  Searching: '{query}' ...", end=" ", flush=True)

        try:
            paths = download_bing_images(query, max_per_query)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        saved = process_and_save(paths, word, word_dir)
        total_new += saved
        print(f"got {len(paths)}, saved {saved}")

        time.sleep(1)

    new_total = existing + total_new
    print(f"  [{word}] Total: {new_total} images (+{total_new} new)")
    return total_new


def main():
    parser = argparse.ArgumentParser(description="Collect close-up training images")
    parser.add_argument("--word", type=str, help="Collect for a specific word")
    parser.add_argument("--max-per-query", type=int, default=40, help="Max images per query")
    args = parser.parse_args()

    if args.word:
        if args.word not in CLOSEUP_QUERIES:
            print(f"No queries for '{args.word}'. Available: {sorted(CLOSEUP_QUERIES)}")
            sys.exit(1)
        collect_for_word(args.word, CLOSEUP_QUERIES[args.word], args.max_per_query)
    else:
        total = 0
        for word, queries in sorted(CLOSEUP_QUERIES.items()):
            total += collect_for_word(word, queries, args.max_per_query)
        print(f"\n=== DONE: {total} new close-up images across {len(CLOSEUP_QUERIES)} words ===")


if __name__ == "__main__":
    main()
