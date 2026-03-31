#!/usr/bin/env python3
"""Collect targeted training data for problem classes identified during device testing.

Three problem domains:
1. Depicted panda (and other depicted animals) — prints on rugs, fabrics, nursery decor
2. Cup in context — cups on desks near laptops, keyboards, electronics
3. Fridge — phone-camera perspective in kitchens

Uses icrawler (Bing) or duckduckgo_search for image collection.

Usage:
    python scripts/collect_problem_class_data.py --all
    python scripts/collect_problem_class_data.py --word cup
    python scripts/collect_problem_class_data.py --word panda
    python scripts/collect_problem_class_data.py --list
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

# ── Problem Class Queries ──────────────────────────────────────────────────
# Each query targets the specific gap between training data and phone camera reality.

PROBLEM_CLASS_QUERIES = {
    # ── Depicted panda: prints, textiles, stuffed animals ──────────────────
    # The model sees real zoo pandas but never 2D depictions on household items
    "panda": [
        # Textile/fabric depictions
        "panda print rug",
        "panda rug nursery",
        "panda blanket baby",
        "panda pillow cushion",
        "panda fabric print",
        "panda pattern textile",
        "panda towel bathroom",
        "panda bedding kids",
        # Stuffed animals (phone camera angle)
        "stuffed panda toy on bed",
        "panda stuffed animal on couch",
        "panda plush toy nursery",
        "stuffed panda close up",
        "panda teddy bear on shelf",
        # Wall art and decor
        "panda wall art nursery",
        "panda poster kids room",
        "panda print framed wall",
        "panda decal nursery wall",
        # Phone camera perspectives of real pandas
        "panda at zoo phone photo",
        "panda exhibit zoo visitor photo",
        "panda behind glass zoo",
    ],

    # ── Cup in context: on desks near electronics ──────────────────────────
    # The model struggles when cup is next to laptop/keyboard because the
    # 270px crop captures surrounding electronics
    "cup": [
        # Cup on desk near electronics (the exact failure mode)
        "coffee cup next to laptop on desk",
        "mug next to laptop keyboard",
        "coffee mug on desk with computer",
        "cup on desk with keyboard and mouse",
        "mug next to macbook on desk",
        "coffee cup work desk computer",
        "cup on table next to laptop",
        "mug between laptop and keyboard",
        # Cup on various surfaces (close up, phone angle)
        "coffee mug on table from above",
        "cup on counter top phone photo",
        "mug on kitchen counter close up",
        "cup on couch armrest",
        "coffee cup on nightstand",
        "mug on coffee table",
        "cup held in hand",
        "travel mug on desk",
        # Small cups that would be small in frame
        "espresso cup on desk",
        "small cup on table setting",
        "tea cup saucer on desk",
        "cup on office desk cluttered",
    ],

    # ── Fridge: phone camera perspective in kitchens ───────────────────────
    # 935 existing images but 0% on-device recognition — all web product shots
    "fridge": [
        # Full fridge views from kitchen
        "refrigerator in kitchen home",
        "fridge kitchen corner",
        "fridge in small kitchen",
        "kitchen with refrigerator side view",
        "refrigerator between cabinets kitchen",
        "fridge standing in kitchen",
        "white fridge kitchen",
        "stainless steel fridge kitchen",
        # Close up and partial views (what you see from arm's length)
        "fridge door close up",
        "fridge with magnets on door",
        "fridge handle close up",
        "refrigerator door magnets photos",
        "fridge door with kids drawings",
        # Different fridge types
        "french door refrigerator kitchen",
        "side by side fridge kitchen",
        "mini fridge room",
        "fridge open door inside",
        "fridge top freezer kitchen",
        # Phone camera angles
        "my fridge kitchen photo",
        "new fridge kitchen instagram",
        "apartment kitchen fridge",
        "fridge next to counter kitchen",
    ],

    # ── Depicted animals (other species) ───────────────────────────────────
    # Same domain gap: model trained on real animals, user has depicted ones
    "lion": [
        "lion print rug",
        "lion pillow cushion",
        "stuffed lion toy on bed",
        "lion wall art nursery",
        "lion picture book page",
        "lion print blanket kids",
    ],
    "elephant": [
        "elephant print rug",
        "elephant pillow nursery",
        "stuffed elephant toy",
        "elephant wall decal nursery",
        "elephant blanket baby",
        "elephant picture book page",
    ],
    "bear": [
        "bear print rug kids",
        "stuffed bear on couch",
        "teddy bear on bed close up",
        "bear wall art nursery",
        "bear pillow cushion kids",
        "bear picture book page",
    ],
    "dog": [
        "dog print rug doormat",
        "stuffed dog toy on bed",
        "dog pillow cushion couch",
        "dog picture book kids",
        "dog pattern fabric",
    ],
    "cat": [
        "cat print rug doormat",
        "stuffed cat toy on couch",
        "cat pillow cushion",
        "cat picture book kids",
        "cat pattern fabric",
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
    import requests

    tmp_dir = Path(tempfile.mkdtemp(prefix="seeky_crawl_"))

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

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        w, h = img.size
        if w < 50 or h < 50:
            continue

        img = img.resize(CROP_SIZE, Image.LANCZOS)

        content_hash = hashlib.md5(img.tobytes()[:10000]).hexdigest()[:8]
        fname = f"{word}_{prefix}_{content_hash}.jpg"
        out_path = word_dir / fname

        if out_path.exists():
            continue

        img.save(out_path, "JPEG", quality=90)
        saved += 1

    return saved


def collect_for_word(word: str, queries: list[str], max_per_query: int = 50, engine: str = "bing"):
    """Collect training images for a word."""
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

        saved = process_and_save(paths, word, word_dir, "prob")
        total_new += saved
        print(f"got {len(paths)}, saved {saved}")

        time.sleep(1)

    new_total = existing + total_new
    print(f"  [{word}] Total: {new_total} images (+{total_new} new)")
    return total_new


def main():
    parser = argparse.ArgumentParser(description="Collect training data for problem classes")
    parser.add_argument("--word", type=str, help="Collect for a specific word")
    parser.add_argument("--all", action="store_true", help="Collect for all problem classes")
    parser.add_argument("--core", action="store_true", help="Collect for core 3 (panda, cup, fridge)")
    parser.add_argument("--max-per-query", type=int, default=50, help="Max images per search query")
    parser.add_argument("--engine", choices=["bing", "ddg"], default="bing",
                        help="Search engine (bing or ddg)")
    parser.add_argument("--list", action="store_true", help="List available words")
    args = parser.parse_args()

    if args.list:
        for word, queries in sorted(PROBLEM_CLASS_QUERIES.items()):
            print(f"  {word}: {len(queries)} queries")
        return

    if args.word:
        if args.word not in PROBLEM_CLASS_QUERIES:
            print(f"No queries defined for '{args.word}'. Use --list to see available words.")
            sys.exit(1)
        collect_for_word(args.word, PROBLEM_CLASS_QUERIES[args.word], args.max_per_query, args.engine)
    elif args.core:
        total = 0
        for word in ["panda", "cup", "fridge"]:
            total += collect_for_word(word, PROBLEM_CLASS_QUERIES[word], args.max_per_query, args.engine)
        print(f"\n=== DONE: {total} new images across core 3 problem classes ===")
    elif args.all:
        total = 0
        for word, queries in sorted(PROBLEM_CLASS_QUERIES.items()):
            total += collect_for_word(word, queries, args.max_per_query, args.engine)
        print(f"\n=== DONE: {total} new images across {len(PROBLEM_CLASS_QUERIES)} words ===")
    else:
        print("Specify --core (panda/cup/fridge), --all, or --word WORD")
        sys.exit(1)


if __name__ == "__main__":
    main()
