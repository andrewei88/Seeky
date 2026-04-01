#!/usr/bin/env python3
"""Collect targeted training data for problem classes identified during device testing.

Problem domains:
1. Depicted panda (and other depicted animals) — prints on rugs, fabrics, nursery decor
2. Cup in context — cups on desks near laptops, keyboards, electronics
3. Fridge — phone-camera perspective in kitchens
4. Shoe — top-down phone angle on floors (v5, 2026-03-31)
5. Toothbrush — in holders/cups in bathrooms, phone-camera angle (v5)
6. Picture — framed pictures on walls, emphasizing frames (v5)
7. Trash can — NEW CLASS, various types in home rooms (v5)
8. Scale — NEW CLASS, bathroom scales on floor, top-down angle (v5)
9. Dumbbell — NEW CLASS, home gym dumbbells on floors/racks (v6, 2026-04-01)
10. Panda (extra) — more depicted panda on flat textiles, rugs, mats (v6)

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

    # ── Shoe: phone-camera top-down angle on floors ──────────────────────
    # Training data is side-profile product shots. Phone reality is looking
    # down at shoes on a floor. Need top-down angles, pairs, various types.
    "shoe": [
        # Top-down angles (the specific gap)
        "shoes on floor top view",
        "shoes on doormat top down",
        "pair of shoes floor overhead",
        "sneakers on floor looking down",
        "shoes at front door top view",
        "shoes on carpet from above",
        "shoes on wooden floor overhead",
        "pair of shoes on tile floor",
        # Kid shoes specifically
        "toddler shoes on floor",
        "baby shoes on carpet",
        "kids sneakers on floor",
        "children shoes at door",
        "small shoes on mat",
        # Various shoe types from natural angles
        "sandals on floor",
        "boots on doormat",
        "slippers on bedroom floor",
        "flip flops on floor",
        "running shoes on floor photo",
        # In context (where kids see shoes)
        "shoes by front door home",
        "shoe rack with shoes",
        "shoes near couch on floor",
        "shoes under bench entryway",
    ],

    # ── Toothbrush: in bathroom context, phone-camera angle ──────────────
    # Small thin object, usually in a holder or cup. Crop captures the
    # counter/sink instead of the toothbrush. Need close-up in-context shots.
    "toothbrush": [
        # In holder/cup (the typical home scenario)
        "toothbrush in holder bathroom",
        "toothbrush in cup on counter",
        "toothbrush holder bathroom counter",
        "toothbrush in glass bathroom sink",
        "toothbrush standing in cup",
        "electric toothbrush on bathroom counter",
        "toothbrush next to sink",
        # Close up phone-camera angles
        "toothbrush close up on counter",
        "toothbrush on bathroom shelf",
        "toothbrush lying on counter",
        "toothbrush on sink edge",
        "manual toothbrush close up",
        # Multiple toothbrushes
        "family toothbrushes in holder",
        "kids toothbrush in cup",
        "colorful toothbrushes in holder",
        # Various types
        "electric toothbrush charging stand",
        "kids toothbrush on counter",
        "bamboo toothbrush on counter",
        "toothbrush and toothpaste on counter",
    ],

    # ── Picture: framed pictures on walls, emphasizing the frame ─────────
    # The model sees the content of the picture, not the frame. Training
    # data needs to emphasize frames as the defining visual feature.
    "picture": [
        # Framed pictures on walls (the primary use case)
        "framed picture on wall",
        "framed photo on wall home",
        "picture frame on wall living room",
        "framed art on wall home",
        "family photo framed on wall",
        "framed picture hanging on wall",
        "picture frame on bedroom wall",
        "framed print on wall",
        # Various frame types
        "wooden picture frame on wall",
        "black picture frame on wall",
        "gold picture frame on wall",
        "white picture frame on wall",
        "ornate picture frame on wall",
        # Gallery walls (multiple framed pictures)
        "gallery wall picture frames",
        "multiple framed pictures on wall",
        "photo wall arrangement home",
        # On desk/shelf (secondary)
        "framed photo on desk",
        "picture frame on shelf",
        "framed photo on nightstand",
        "photo frame on table",
        # Close up phone angle
        "picture frame close up on wall",
        "framed photo close up home",
    ],

    # ── Trash can: NEW CLASS — various types in home rooms ───────────────
    # Not in the 154-class model. Needs diverse examples: kitchen tall bin,
    # bathroom small bin, office wastebasket, pedal/step-on, with/without lid.
    "trash can": [
        # Kitchen trash cans
        "kitchen trash can",
        "kitchen garbage can stainless steel",
        "step on trash can kitchen",
        "pedal trash can kitchen",
        "tall trash can kitchen corner",
        "kitchen trash can next to counter",
        "trash can in kitchen home",
        # Bathroom trash cans
        "small trash can bathroom",
        "bathroom wastebasket",
        "bathroom trash can next to toilet",
        "small waste bin bathroom",
        "bathroom garbage can",
        # Office/bedroom
        "office wastebasket",
        "desk trash can",
        "bedroom trash can",
        "mesh wastebasket office",
        "small trash can bedroom",
        # Various types
        "trash can with lid home",
        "open top trash can",
        "plastic trash can home",
        "stainless steel trash can",
        "white trash can home",
        "trash can with foot pedal",
        # Phone camera angles
        "my trash can home photo",
        "trash can on floor home",
        "garbage can in room",
        "recycling bin kitchen",
    ],

    # ── Dumbbell: NEW CLASS — home gym dumbbells ──────────────────────────
    # Kids see these at home on floors, on racks, on mats. Distinctive shape
    # (two weights connected by a bar). Need variety: neoprene, rubber, metal,
    # adjustable, and kid toy weights.
    "dumbbell": [
        # On floor (the typical home view)
        "dumbbell on floor",
        "dumbbells on gym floor",
        "dumbbell on yoga mat",
        "dumbbells on carpet home",
        "pair of dumbbells on floor",
        "single dumbbell on floor close up",
        "dumbbell on hardwood floor",
        "dumbbells on exercise mat",
        # On rack/stand
        "dumbbell rack home gym",
        "dumbbells on rack",
        "dumbbell set on stand",
        "dumbbell rack close up",
        # Various types
        "neoprene dumbbells colorful",
        "rubber hex dumbbells",
        "adjustable dumbbell home",
        "metal dumbbell chrome",
        "small hand weights exercise",
        "colorful dumbbells set",
        "cast iron dumbbell",
        "vinyl coated dumbbells",
        # Phone camera angles
        "dumbbell close up photo",
        "dumbbells home workout photo",
        "my dumbbells home gym",
        "dumbbell next to water bottle",
        "dumbbells on bench home",
        "dumbbell in hand workout",
        "kids toy dumbbell",
        "toddler toy weights",
    ],

    # ── Panda (extra round): flat textiles, rugs, mats ──────────────────
    # Previous prob data (1083 images) wasn't enough to overcome the domain
    # gap. Model still confuses panda rugs with cookies (flat, textured).
    # Need more specifically flat/textile panda depictions.
    "panda_extra": [
        # Flat panda rugs and mats (the exact failure mode)
        "panda rug on floor",
        "panda mat on floor close up",
        "panda face rug nursery",
        "panda area rug kids room",
        "round panda rug",
        "panda doormat",
        "panda bath mat",
        "panda floor mat close up",
        # Panda prints on flat surfaces
        "panda print on wall close up",
        "panda poster on wall phone photo",
        "panda picture on wall nursery",
        "panda artwork flat",
        "panda face illustration print",
        "panda painting nursery wall",
        # Panda on fabric (flat/draped)
        "panda blanket flat on bed",
        "panda towel flat",
        "panda cushion cover flat",
        "panda print fabric close up",
        "panda tapestry wall hanging",
        "panda throw pillow flat on couch",
    ],

    # ── Scale: NEW CLASS — bathroom scales, top-down angle ───────────────
    # Not in the 154-class model. Phone-camera angle is looking straight
    # down at a flat rectangle on a bathroom floor. Confusable with laptop,
    # book, or mat. Need distinctive scale features (display, glass surface).
    "scale": [
        # Bathroom scale on floor (the typical view)
        "bathroom scale on floor",
        "digital bathroom scale on tile",
        "body weight scale bathroom floor",
        "bathroom scale on bathroom floor",
        "scale on tile floor bathroom",
        "weight scale on floor",
        "body scale bathroom",
        # Top-down angles
        "bathroom scale top view",
        "standing on bathroom scale",
        "feet on bathroom scale",
        "bathroom scale from above",
        "digital scale display bathroom",
        # Various types
        "glass bathroom scale",
        "digital bathroom scale",
        "smart body scale",
        "white bathroom scale",
        "black bathroom scale on floor",
        "analog bathroom scale",
        "round bathroom scale",
        # In context
        "bathroom scale next to bathtub",
        "scale on bathroom mat",
        "bathroom floor with scale",
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


# Map query keys to filesystem class names (when they differ)
QUERY_TO_CLASS = {
    "panda_extra": "panda",
}


def collect_for_word(word: str, queries: list[str], max_per_query: int = 50, engine: str = "bing"):
    """Collect training images for a word."""
    class_name = QUERY_TO_CLASS.get(word, word)
    word_dir = OUTPUT_DIR / class_name.replace(" ", "_")
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

        saved = process_and_save(paths, class_name, word_dir, "prob")
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
    parser.add_argument("--v5", action="store_true", help="Collect for v5 round (shoe, toothbrush, picture, trash can, scale)")
    parser.add_argument("--v6", action="store_true", help="Collect for v6 round (dumbbell + extra panda)")
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
    elif args.v6:
        total = 0
        for word in ["dumbbell", "panda_extra"]:
            total += collect_for_word(word, PROBLEM_CLASS_QUERIES[word], args.max_per_query, args.engine)
        print(f"\n=== DONE: {total} new images across v6 classes (dumbbell + panda extra) ===")
    elif args.v5:
        total = 0
        for word in ["shoe", "toothbrush", "picture", "trash can", "scale"]:
            total += collect_for_word(word, PROBLEM_CLASS_QUERIES[word], args.max_per_query, args.engine)
        print(f"\n=== DONE: {total} new images across 5 v5 classes ===")
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
