#!/usr/bin/env python3
"""Collect phone-camera-like training images for home objects.

The existing training data is mostly product shots from web scraping.
Real-world phone camera images look very different: objects in home context,
room lighting, arm's-length angles, partial views.

This script targets search queries that produce in-context photos:
  - "bluetooth speaker on desk"
  - "smart speaker living room"
  - "bookshelf home office"

Uses Bing Image Search API (or fallback to icrawler/DuckDuckGo).

Usage:
    python scripts/collect_home_context_data.py [--word WORD] [--max-per-query 50]
    python scripts/collect_home_context_data.py --all
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "arya_training" / "all"
CROP_SIZE = (256, 256)

# Priority home objects with targeted search queries.
# Each query is designed to return images that look like phone camera photos:
# - Objects in room context (not isolated product shots)
# - Various angles and distances
# - Home lighting conditions
HOME_OBJECT_QUERIES = {
    "speaker": [
        "bluetooth speaker on desk",
        "smart speaker living room",
        "JBL speaker on table",
        "speaker on shelf home",
        "portable speaker kitchen counter",
        "wireless speaker bedroom nightstand",
        "sonos speaker room",
        "homepod on counter",
        "alexa echo on table",
        "bose speaker home",
        "small speaker on bookshelf",
        "speaker next to laptop",
    ],
    "shelf": [
        "bookshelf living room",
        "shelf with books home",
        "floating shelf wall",
        "kitchen shelf",
        "bathroom shelf towels",
        "shelves home office",
        "wall shelf decorations",
        "open shelving kitchen",
    ],
    "tv": [
        "tv on wall living room",
        "television stand home",
        "flat screen tv mounted",
        "tv in bedroom",
        "smart tv living room",
        "tv screen home",
    ],
    "laptop": [
        "laptop on desk",
        "laptop on table home",
        "laptop open on couch",
        "laptop kitchen counter",
        "macbook on desk",
        "laptop home office",
        "laptop from above",
        "laptop side angle desk",
    ],
    "keyboard": [
        "keyboard on desk",
        "computer keyboard home office",
        "wireless keyboard desk",
        "keyboard and mouse setup",
    ],
    "microwave": [
        "microwave in kitchen",
        "microwave on counter",
        "microwave kitchen corner",
    ],
    "toaster": [
        "toaster on kitchen counter",
        "toaster in kitchen",
        "toaster bread kitchen",
    ],
    "fridge": [
        "refrigerator kitchen",
        "fridge in kitchen",
        "fridge door open kitchen",
    ],
    "oven": [
        "oven in kitchen",
        "oven door kitchen",
        "stove oven kitchen",
    ],
    "couch": [
        "couch living room",
        "sofa in living room",
        "couch from above",
        "couch cushions home",
    ],
    "chair": [
        "chair at desk",
        "dining chair table",
        "office chair home",
        "kitchen chair",
    ],
    "table": [
        "dining table home",
        "kitchen table",
        "coffee table living room",
        "desk table home office",
    ],
    "light": [
        "ceiling light room",
        "lamp on table",
        "light fixture home",
        "pendant light kitchen",
        "floor lamp living room",
        "desk lamp home office",
    ],
    "door": [
        "door in house",
        "front door interior",
        "bedroom door open",
        "door hallway home",
    ],
    "window": [
        "window inside home",
        "window living room",
        "bedroom window",
        "kitchen window",
    ],
    "cup": [
        "coffee cup on table",
        "cup on kitchen counter",
        "mug on desk",
        "cup next to laptop",
    ],
    "bottle": [
        "water bottle on desk",
        "bottle on table",
        "water bottle kitchen counter",
    ],
    "remote": [
        "tv remote on couch",
        "remote control on table",
        "remote coffee table",
    ],
    "phone": [
        "phone on table",
        "smartphone on desk",
        "phone on kitchen counter",
        "phone charging on nightstand",
    ],
    "clock": [
        "wall clock home",
        "clock on wall living room",
        "alarm clock nightstand",
    ],
    "fan": [
        "ceiling fan room",
        "desk fan home",
        "standing fan living room",
    ],
    "mirror": [
        "mirror on wall bathroom",
        "mirror bedroom",
        "wall mirror home",
    ],
    "towel": [
        "towel hanging bathroom",
        "towel rack bathroom",
        "kitchen towel hanging",
    ],
    "pillow": [
        "pillow on bed",
        "pillows on couch",
        "throw pillow sofa",
    ],
    "blanket": [
        "blanket on bed",
        "blanket on couch",
        "throw blanket sofa",
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

    tmp_dir = Path(tempfile.mkdtemp(prefix="arya_crawl_"))

    crawler = BingImageCrawler(
        storage={"root_dir": str(tmp_dir)},
        log_level="WARNING",
    )
    crawler.crawl(keyword=query, max_num=max_images)

    return list(tmp_dir.glob("*"))


def download_duckduckgo_images(query: str, max_images: int = 50) -> list[Path]:
    """Download images from DuckDuckGo using duckduckgo_search."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("  pip install duckduckgo_search  (required for DuckDuckGo image search)")
        sys.exit(1)

    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="arya_crawl_"))

    import requests

    with DDGS() as ddgs:
        results = list(ddgs.images(query, max_results=max_images))

    downloaded = []
    for i, result in enumerate(results):
        url = result.get("image", "")
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200 and len(resp.content) > 1000:
                ext = ".jpg"
                if "png" in resp.headers.get("content-type", ""):
                    ext = ".png"
                path = tmp_dir / f"img_{i:04d}{ext}"
                path.write_bytes(resp.content)
                downloaded.append(path)
        except Exception:
            continue

    return downloaded


def process_and_save(image_paths: list[Path], word: str, word_dir: Path, prefix: str) -> int:
    """Process downloaded images and save as training data."""
    saved = 0
    existing = len(list(word_dir.glob("*.jpg")))

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
        fname = f"{word}_{prefix}_{content_hash}.jpg"
        out_path = word_dir / fname

        if out_path.exists():
            continue

        img.save(out_path, "JPEG", quality=90)
        saved += 1

    return saved


def collect_for_word(word: str, queries: list[str], max_per_query: int = 50, engine: str = "bing"):
    """Collect in-context images for a word."""
    word_dir = OUTPUT_DIR / word.replace(" ", "_")
    word_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(word_dir.glob("*.jpg")))
    print(f"\n[{word}] Existing: {existing} images")

    total_new = 0
    for query in queries:
        print(f"  Searching: '{query}' ...", end=" ", flush=True)

        try:
            if engine == "ddg":
                paths = download_duckduckgo_images(query, max_per_query)
            else:
                paths = download_bing_images(query, max_per_query)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        saved = process_and_save(paths, word, word_dir, "ctx")
        total_new += saved
        print(f"got {len(paths)}, saved {saved}")

        # Rate limit
        time.sleep(1)

    new_total = existing + total_new
    print(f"  [{word}] Total: {new_total} images (+{total_new} new)")
    return total_new


def main():
    parser = argparse.ArgumentParser(description="Collect phone-camera-like training images")
    parser.add_argument("--word", type=str, help="Collect for a specific word")
    parser.add_argument("--all", action="store_true", help="Collect for all home objects")
    parser.add_argument("--max-per-query", type=int, default=50, help="Max images per search query")
    parser.add_argument("--engine", choices=["bing", "ddg"], default="bing",
                        help="Search engine (bing or ddg)")
    parser.add_argument("--list", action="store_true", help="List available words")
    args = parser.parse_args()

    if args.list:
        for word, queries in sorted(HOME_OBJECT_QUERIES.items()):
            print(f"  {word}: {len(queries)} queries")
        return

    if args.word:
        if args.word not in HOME_OBJECT_QUERIES:
            print(f"No queries defined for '{args.word}'. Available words:")
            for w in sorted(HOME_OBJECT_QUERIES):
                print(f"  {w}")
            sys.exit(1)
        collect_for_word(args.word, HOME_OBJECT_QUERIES[args.word], args.max_per_query, args.engine)
    elif args.all:
        total = 0
        for word, queries in sorted(HOME_OBJECT_QUERIES.items()):
            total += collect_for_word(word, queries, args.max_per_query, args.engine)
        print(f"\n=== DONE: {total} new images across {len(HOME_OBJECT_QUERIES)} words ===")
    else:
        # Default: collect for the most critical home objects first
        priority = ["speaker", "shelf", "tv", "laptop", "keyboard", "light",
                     "microwave", "toaster", "fridge", "oven", "couch", "chair"]
        total = 0
        for word in priority:
            if word in HOME_OBJECT_QUERIES:
                total += collect_for_word(word, HOME_OBJECT_QUERIES[word], args.max_per_query, args.engine)
        print(f"\n=== DONE: {total} new images across {len(priority)} priority words ===")
        print("Run with --all for all home objects")


if __name__ == "__main__":
    main()
