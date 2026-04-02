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
OUTPUT_DIR = PROJECT_ROOT / "data" / "seeky_training" / "all"
CROP_SIZE = (256, 256)

# Priority home objects with targeted search queries.
# Each query is designed to return images that look like phone camera photos:
# - Objects in room context (not isolated product shots)
# - Various angles and distances
# - Home lighting conditions
HOME_OBJECT_QUERIES = {
    # ── Priority: 16 weak household classes (below 85% val accuracy) ──────
    # Queries designed to return phone-camera-perspective images:
    # objects in rooms, various angles, home lighting, partial views.
    "shelf": [
        "bookshelf living room",
        "shelf with books home",
        "floating shelf wall",
        "kitchen shelf",
        "bathroom shelf towels",
        "shelves home office",
        "wall shelf decorations",
        "open shelving kitchen",
        "wooden shelf with plants",
        "closet shelf clothes",
        "garage shelf storage",
        "shelf above desk",
    ],
    "table": [
        "dining table home",
        "kitchen table with food",
        "coffee table living room",
        "desk table home office",
        "dining table set for dinner",
        "wooden table close up",
        "table with plates and cups",
        "kitchen table from above",
        "side table lamp nightstand",
        "table legs wooden floor",
    ],
    "blanket": [
        "blanket on bed",
        "blanket on couch",
        "throw blanket sofa",
        "folded blanket on chair",
        "blanket draped over couch arm",
        "knit blanket on bed",
        "baby blanket crib",
        "fleece blanket couch",
    ],
    "couch": [
        "couch living room",
        "sofa in living room",
        "couch from above",
        "couch cushions home",
        "couch with pillows",
        "sectional sofa living room",
        "leather couch room",
        "couch armrest close up",
        "sofa side angle room",
    ],
    "mirror": [
        "mirror on wall bathroom",
        "mirror bedroom",
        "wall mirror home",
        "mirror reflection room",
        "bathroom vanity mirror",
        "full length mirror bedroom",
        "round mirror on wall",
        "mirror above dresser",
    ],
    "window": [
        "window inside home",
        "window living room",
        "bedroom window",
        "kitchen window",
        "window with curtains room",
        "window sunlight room",
        "window blinds home",
        "window from inside looking out",
    ],
    "pillow": [
        "pillow on bed",
        "pillows on couch",
        "throw pillow sofa",
        "decorative pillow on chair",
        "pillow on floor",
        "bed pillows close up",
        "couch cushion pillow",
        "pillow pile on bed",
    ],
    "keyboard": [
        "keyboard on desk",
        "computer keyboard home office",
        "wireless keyboard desk",
        "keyboard and mouse setup",
        "laptop keyboard close up",
        "keyboard from above desk",
        "mechanical keyboard desk",
        "keyboard next to monitor",
    ],
    "book": [
        "book on table",
        "book on nightstand",
        "open book on desk",
        "stack of books on shelf",
        "children's book on floor",
        "book on couch",
        "book next to coffee cup",
        "book on bed",
        "reading book in hand",
        "book on kitchen counter",
    ],
    "towel": [
        "towel hanging bathroom",
        "towel rack bathroom",
        "kitchen towel hanging",
        "towel on hook bathroom",
        "folded towels shelf",
        "bath towel hanging door",
        "hand towel bathroom sink",
        "towel bar bathroom wall",
    ],
    "spoon": [
        "spoon on table",
        "spoon in bowl",
        "spoon next to plate",
        "wooden spoon kitchen counter",
        "spoon in mug",
        "baby spoon high chair",
        "spoon on napkin",
        "spoons in drawer",
        "spoon cereal bowl",
        "measuring spoons kitchen",
    ],
    "soap": [
        "soap dispenser bathroom sink",
        "bar of soap bathroom",
        "hand soap sink counter",
        "soap dish bathroom",
        "liquid soap pump bottle bathroom",
        "soap on bathroom shelf",
        "dish soap kitchen sink",
        "soap bar shower",
    ],
    "bottle": [
        "water bottle on desk",
        "bottle on table",
        "water bottle kitchen counter",
        "baby bottle on counter",
        "bottle on nightstand",
        "shampoo bottle bathroom",
        "water bottle next to laptop",
        "bottle on dining table",
    ],
    "chair": [
        "chair at desk",
        "dining chair table",
        "office chair home",
        "kitchen chair",
        "chair in living room",
        "wooden chair dining room",
        "desk chair from behind",
        "high chair kitchen",
        "rocking chair nursery",
    ],
    "bowl": [
        "bowl on table",
        "cereal bowl on counter",
        "bowl of soup on table",
        "fruit bowl kitchen counter",
        "bowl on dining table",
        "mixing bowl kitchen",
        "bowl next to spoon",
        "empty bowl on table",
        "salad bowl on counter",
        "pet bowl on floor",
    ],
    "door": [
        "door in house",
        "front door interior",
        "bedroom door open",
        "door hallway home",
        "bathroom door",
        "closet door",
        "door handle close up home",
        "open door into room",
    ],
    # ── Other home objects (already had queries, keeping as-is) ───────────
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
    "light": [
        "ceiling light room",
        "lamp on table",
        "light fixture home",
        "pendant light kitchen",
        "floor lamp living room",
        "desk lamp home office",
    ],
    "cup": [
        "coffee cup on table",
        "cup on kitchen counter",
        "mug on desk",
        "cup next to laptop",
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
}


def download_bing_images(query: str, max_images: int = 50) -> list[Path]:
    """Download images from Bing Image Search using icrawler."""
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("  pip install icrawler  (required for Bing image search)")
        sys.exit(1)

    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="seeky_crawl_"))

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

    tmp_dir = Path(tempfile.mkdtemp(prefix="seeky_crawl_"))

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
        # Default: collect for the 16 weakest household classes (below 85% val accuracy)
        priority = ["shelf", "table", "blanket", "couch", "mirror", "window",
                     "pillow", "keyboard", "book", "towel", "spoon", "soap",
                     "bottle", "chair", "bowl", "door"]
        total = 0
        for word in priority:
            if word in HOME_OBJECT_QUERIES:
                total += collect_for_word(word, HOME_OBJECT_QUERIES[word], args.max_per_query, args.engine)
        print(f"\n=== DONE: {total} new images across {len(priority)} priority words ===")
        print("Run with --all for all home objects")


if __name__ == "__main__":
    main()
