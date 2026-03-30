#!/usr/bin/env python3
"""Collect phone-camera-perspective training images for animals and fridge.

Problem: Existing training data is web-scraped (product shots, wildlife
photography). The model sees objects through a phone camera in homes, zoos,
and aquariums — completely different distribution. Fridge has 935 images but
0% on-device recognition. Animals work on web val but fail in real rooms.

This script collects three types of images per class:
  1. Phone-perspective: object as seen through a phone camera in real context
  2. Close-up: object filling the frame (good feature learning)
  3. Depicted: toys, book illustrations, fabric prints, figurines
     (a lion in a book IS a lion to a toddler learning the word)

Usage:
    python scripts/collect_animal_and_fridge_data.py                    # all categories
    python scripts/collect_animal_and_fridge_data.py --word dog         # single word
    python scripts/collect_animal_and_fridge_data.py --category zoo     # one category
    python scripts/collect_animal_and_fridge_data.py --list             # show all words
    python scripts/collect_animal_and_fridge_data.py --engine ddg       # use DuckDuckGo
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "seeky_training" / "all"
CROP_SIZE = (256, 256)


# ── Query definitions ────────────────────────────────────────────────────────
# Each word maps to a list of search queries. Queries are designed to return
# images that match the phone-camera distribution: real context, home lighting,
# arm's-length angles, zoo/aquarium glass, depicted on toys/books/fabric.
#
# Query design principles:
#   - "phone photo of X" and "iphone photo X" return amateur phone shots
#   - "X in living room" / "X on couch" return home-context images
#   - "X at zoo" / "X aquarium" return through-glass phone photos
#   - "X toy" / "X children's book" / "X on blanket" return depicted versions
#   - "X from above" / "X close up" fill perspective gaps
# ─────────────────────────────────────────────────────────────────────────────

FRIDGE_QUERIES = {
    "fridge": [
        # Phone-perspective: fridge in kitchen from standing distance
        "fridge in kitchen",
        "refrigerator in kitchen room",
        "fridge kitchen phone photo",
        "fridge from across kitchen",
        "stainless steel fridge kitchen",
        "white fridge in kitchen",
        "black fridge kitchen",
        "fridge with magnets kitchen",
        "fridge photos on fridge door kitchen",
        "fridge next to counter kitchen",
        "fridge between cabinets kitchen",
        "apartment kitchen fridge",
        "small kitchen with fridge",
        "fridge side by side kitchen",
        "fridge french door kitchen",
        "mini fridge in room",
        "fridge door closed kitchen",
        "refrigerator door kitchen home",
        "my kitchen fridge",
        "new fridge in kitchen",
        # Close-up: fridge door surface, handles
        "fridge door close up",
        "fridge handle close up",
        "fridge magnets close up",
        "refrigerator door stainless steel close up",
        # Context variations
        "fridge in garage",
        "fridge in basement",
        "fridge in office break room",
        "retro fridge kitchen",
        "fridge with water dispenser kitchen",
        "fridge partially visible kitchen corner",
    ],
}

COMMON_ANIMAL_QUERIES = {
    "dog": [
        # Phone photos in home context
        "dog on couch phone photo",
        "dog lying on floor living room",
        "dog on bed",
        "dog in backyard phone",
        "dog sitting on carpet",
        "dog looking at camera phone",
        "puppy on couch",
        "dog in kitchen home",
        "dog on stairs home",
        "dog from above on floor",
        "small dog on lap",
        "dog next to person on couch",
        "dog sleeping on floor home",
        "dog playing in yard phone photo",
        # Close-up
        "dog face close up phone",
        "puppy close up",
        "dog portrait phone camera",
        # Depicted
        "stuffed dog toy plush",
        "dog in children's book illustration",
        "dog figurine toy",
        "dog picture on wall",
        "dog print on blanket",
        "toy dog stuffed animal",
    ],
    "cat": [
        # Phone photos in home context
        "cat on couch phone photo",
        "cat on bed sleeping",
        "cat on windowsill",
        "cat on carpet floor",
        "cat in living room home",
        "cat lying on chair",
        "kitten on couch",
        "cat looking at camera phone",
        "cat from above on floor",
        "cat on kitchen counter",
        "cat sitting on table",
        "cat on lap phone photo",
        "cat sleeping on bed home",
        "cat on stairs home",
        # Close-up
        "cat face close up phone",
        "kitten close up",
        "cat portrait phone camera",
        # Depicted
        "stuffed cat toy plush",
        "cat in children's book illustration",
        "cat figurine toy",
        "cat picture on wall",
        "cat print on blanket",
        "toy cat stuffed animal",
    ],
    "bird": [
        # Phone photos in real context
        "bird on fence phone photo",
        "bird in backyard phone",
        "bird at bird feeder",
        "bird on tree branch close",
        "bird on ground yard",
        "bird on porch railing",
        "bird at window home",
        "bird in park phone photo",
        "small bird on fence",
        "bird on power line from below",
        "bird in cage home",
        "pet bird on shoulder",
        "parrot in cage home",
        # Close-up
        "bird close up phone photo",
        "small bird close up",
        "bird perched close up",
        # Depicted
        "bird toy figurine",
        "bird in children's book illustration",
        "bird stuffed animal plush",
        "bird picture wall art",
        "bird on baby blanket print",
        "toy bird for kids",
    ],
    "squirrel": [
        # Phone photos in real context
        "squirrel in backyard phone photo",
        "squirrel on tree phone",
        "squirrel on fence backyard",
        "squirrel in park phone photo",
        "squirrel eating nut close",
        "squirrel on porch",
        "squirrel from window home",
        "squirrel on bird feeder",
        "squirrel on ground yard",
        "squirrel climbing tree phone",
        # Close-up
        "squirrel close up phone",
        "squirrel face close up",
        # Depicted
        "squirrel stuffed animal toy",
        "squirrel in children's book illustration",
        "squirrel figurine toy",
        "squirrel picture wall art",
    ],
}

ZOO_ANIMAL_QUERIES = {
    "elephant": [
        # Phone photos from zoo
        "elephant at zoo phone photo",
        "elephant zoo enclosure",
        "elephant at zoo from viewing area",
        "elephant behind fence zoo",
        "baby elephant zoo",
        "elephant walking zoo",
        "elephant zoo visitor photo",
        "elephant spraying water zoo",
        "elephant eating zoo",
        "elephant family zoo",
        # Close-up
        "elephant close up zoo",
        "elephant face close up",
        "elephant trunk close up",
        # Depicted (important: toddlers see these at home)
        "elephant stuffed animal toy",
        "elephant toy figurine",
        "elephant in children's book illustration",
        "elephant on baby blanket",
        "elephant print fabric nursery",
        "elephant plush toy on couch",
        "elephant picture nursery wall",
        "elephant toy on floor",
        "elephant rubber bath toy",
        "elephant on rug nursery",
    ],
    "giraffe": [
        # Phone photos from zoo
        "giraffe at zoo phone photo",
        "giraffe zoo enclosure",
        "giraffe zoo visitor photo",
        "giraffe feeding zoo",
        "giraffe behind fence zoo",
        "giraffe at zoo from below",
        "baby giraffe zoo",
        "giraffe neck head zoo close",
        # Close-up
        "giraffe close up zoo",
        "giraffe face close up",
        # Depicted
        "giraffe stuffed animal toy",
        "giraffe toy figurine",
        "giraffe in children's book illustration",
        "giraffe on baby blanket",
        "giraffe plush toy",
        "giraffe picture nursery wall art",
        "giraffe on fabric print",
        "giraffe rubber toy",
    ],
    "zebra": [
        # Phone photos from zoo
        "zebra at zoo phone photo",
        "zebra zoo enclosure",
        "zebra zoo visitor photo",
        "zebra behind fence zoo",
        "zebra herd zoo",
        "zebra grazing zoo",
        # Close-up
        "zebra close up zoo",
        "zebra stripes close up",
        "zebra face close up",
        # Depicted
        "zebra stuffed animal toy",
        "zebra toy figurine",
        "zebra in children's book illustration",
        "zebra print fabric nursery",
        "zebra plush toy",
        "zebra on baby blanket",
        "zebra picture wall art nursery",
    ],
    "lion": [
        # Phone photos from zoo
        "lion at zoo phone photo",
        "lion zoo enclosure",
        "lion behind glass zoo",
        "lion sleeping zoo",
        "lion zoo visitor photo",
        "lion and lioness zoo",
        "lion walking zoo",
        # Close-up
        "lion close up zoo",
        "lion face close up",
        "lion mane close up",
        # Depicted (very common in nurseries)
        "lion stuffed animal toy",
        "lion toy figurine",
        "lion in children's book illustration",
        "lion on baby blanket nursery",
        "lion plush toy on couch",
        "lion picture nursery wall art",
        "lion print fabric",
        "lion rubber toy",
        "lion on rug pattern",
        "lion cub plush toy",
    ],
    "tiger": [
        # Phone photos from zoo
        "tiger at zoo phone photo",
        "tiger zoo enclosure",
        "tiger behind glass zoo",
        "tiger zoo visitor photo",
        "tiger walking zoo",
        "tiger sleeping zoo",
        "tiger swimming zoo",
        # Close-up
        "tiger close up zoo",
        "tiger face close up",
        "tiger stripes close up",
        # Depicted
        "tiger stuffed animal toy",
        "tiger toy figurine",
        "tiger in children's book illustration",
        "tiger plush toy",
        "tiger on blanket print",
        "tiger picture nursery wall art",
        "tiger rubber toy",
    ],
}

AQUARIUM_ANIMAL_QUERIES = {
    "whale": [
        # Phone photos from aquarium/whale watching
        "whale at aquarium phone photo",
        "whale aquarium tank",
        "whale watching phone photo",
        "whale breach ocean phone",
        "beluga whale aquarium",
        "whale tail ocean phone",
        "whale from boat phone photo",
        "orca killer whale aquarium",
        # Close-up
        "whale close up aquarium",
        "whale underwater close up",
        # Depicted (very common in nurseries)
        "whale stuffed animal toy",
        "whale toy figurine",
        "whale in children's book illustration",
        "whale on baby blanket print",
        "whale plush toy",
        "whale bath toy",
        "whale nursery wall art",
        "whale on fabric print",
        "blue whale toy",
    ],
    "seal": [
        # Phone photos from aquarium/beach
        "seal at aquarium phone photo",
        "seal aquarium tank",
        "seal at beach phone photo",
        "seal on rock beach",
        "seal swimming aquarium",
        "sea lion aquarium show",
        "seal pup beach phone",
        "seal zoo phone photo",
        # Close-up
        "seal close up aquarium",
        "seal face close up",
        # Depicted
        "seal stuffed animal toy",
        "seal toy figurine",
        "seal in children's book illustration",
        "seal plush toy",
        "seal bath toy",
    ],
    "shark": [
        # Phone photos from aquarium
        "shark at aquarium phone photo",
        "shark aquarium tank",
        "shark swimming aquarium glass",
        "shark aquarium tunnel",
        "shark from below aquarium",
        "nurse shark aquarium",
        "reef shark aquarium",
        "hammerhead shark aquarium",
        # Close-up
        "shark close up aquarium",
        "shark face close up",
        # Depicted (common toys)
        "shark stuffed animal toy",
        "shark toy figurine",
        "shark in children's book illustration",
        "shark plush toy",
        "shark bath toy",
        "baby shark toy",
        "shark on blanket print",
    ],
    "jellyfish": [
        # Phone photos from aquarium
        "jellyfish at aquarium phone photo",
        "jellyfish aquarium tank",
        "jellyfish glowing aquarium",
        "jellyfish exhibit aquarium",
        "jellyfish aquarium dark background",
        "moon jellyfish aquarium",
        "jellyfish aquarium close up phone",
        # Close-up
        "jellyfish close up aquarium",
        "jellyfish tentacles close up",
        # Depicted
        "jellyfish toy figurine",
        "jellyfish in children's book illustration",
        "jellyfish plush toy",
        "jellyfish bath toy",
        "jellyfish nursery wall art",
    ],
    "octopus": [
        # Phone photos from aquarium
        "octopus at aquarium phone photo",
        "octopus aquarium tank",
        "octopus aquarium glass",
        "octopus in tank close up",
        "octopus swimming aquarium",
        "giant pacific octopus aquarium",
        # Close-up
        "octopus close up aquarium",
        "octopus tentacles close up",
        # Depicted (popular toys)
        "octopus stuffed animal toy",
        "octopus toy figurine",
        "octopus in children's book illustration",
        "octopus plush toy",
        "octopus bath toy",
        "octopus on blanket print nursery",
        "reversible octopus plush toy",
    ],
    "dolphin": [
        # Phone photos from aquarium/ocean
        "dolphin at aquarium phone photo",
        "dolphin aquarium show",
        "dolphin jumping aquarium",
        "dolphin swimming aquarium glass",
        "dolphin ocean phone photo",
        "dolphin watching from boat phone",
        # Close-up
        "dolphin close up aquarium",
        "dolphin face close up",
        # Depicted
        "dolphin stuffed animal toy",
        "dolphin toy figurine",
        "dolphin in children's book illustration",
        "dolphin plush toy",
        "dolphin bath toy",
    ],
}

# Other animals that need phone-camera and depicted data
OTHER_ANIMAL_QUERIES = {
    "panda": [
        # Depicted (primary use case: panda rug, toys, books)
        "panda on rug floor",
        "panda rug nursery",
        "panda pattern rug",
        "panda blanket print",
        "panda on fabric",
        "panda illustration children's book page",
        "panda in picture book",
        "panda wall art nursery",
        "panda print on shirt",
        "panda face on pillow",
        "panda bath toy",
        "panda rubber toy",
        "panda figurine toy",
        "panda wooden toy",
        # Phone photos from zoo
        "panda at zoo phone photo",
        "panda zoo enclosure",
        "panda eating bamboo zoo phone",
        "panda zoo visitor photo",
        "red panda zoo phone",
        # Already have lots of plush toy data, add more context variation
        "panda toy on floor home",
        "panda plush on couch in living room",
        "panda toy next to other toys",
        "small panda toy on table",
    ],
    "bear": [
        "bear at zoo phone photo",
        "bear zoo enclosure",
        "grizzly bear zoo",
        "polar bear zoo phone photo",
        "bear behind glass zoo",
        "bear close up zoo",
        # Depicted
        "bear stuffed animal toy",
        "bear in children's book illustration",
        "bear plush toy on bed",
        "bear figurine toy",
        "bear on baby blanket print",
        "teddy bear different from bear toy",
    ],
    "hippo": [
        "hippo at zoo phone photo",
        "hippo zoo water enclosure",
        "hippo zoo visitor photo",
        "baby hippo zoo",
        "hippo close up zoo",
        # Depicted
        "hippo stuffed animal toy",
        "hippo in children's book illustration",
        "hippo plush toy",
        "hippo bath toy",
        "hippo figurine toy",
    ],
    "monkey": [
        "monkey at zoo phone photo",
        "monkey zoo enclosure",
        "monkey zoo visitor photo",
        "monkey climbing zoo",
        "monkey behind glass zoo",
        "monkey close up zoo",
        # Depicted
        "monkey stuffed animal toy",
        "monkey in children's book illustration",
        "monkey plush toy",
        "monkey figurine toy",
        "monkey toy on shelf",
    ],
    "penguin": [
        "penguin at zoo phone photo",
        "penguin aquarium",
        "penguin exhibit zoo",
        "penguin swimming aquarium glass",
        "penguin walking zoo",
        "penguin close up zoo",
        # Depicted
        "penguin stuffed animal toy",
        "penguin in children's book illustration",
        "penguin plush toy",
        "penguin figurine toy",
        "penguin bath toy",
        "penguin on blanket print nursery",
    ],
    "owl": [
        "owl at zoo phone photo",
        "owl in tree phone photo",
        "owl zoo exhibit",
        "owl close up",
        "barn owl close up",
        # Depicted
        "owl stuffed animal toy",
        "owl in children's book illustration",
        "owl figurine toy",
        "owl nursery wall art",
        "owl plush toy",
    ],
    "turtle": [
        "turtle at aquarium phone photo",
        "sea turtle aquarium",
        "turtle in backyard phone photo",
        "pet turtle home",
        "turtle at zoo",
        "turtle close up phone",
        # Depicted
        "turtle stuffed animal toy",
        "turtle in children's book illustration",
        "turtle figurine toy",
        "turtle bath toy",
        "turtle plush toy",
    ],
    "fox": [
        "fox in yard phone photo",
        "fox in park phone photo",
        "fox at zoo phone",
        "fox in backyard",
        "fox at zoo enclosure",
        "fox close up",
        # Depicted
        "fox stuffed animal toy",
        "fox in children's book illustration",
        "fox figurine toy",
        "fox plush toy",
        "fox nursery wall art",
    ],
    "otter": [
        # Phone photos from zoo/aquarium
        "otter at zoo phone photo",
        "otter aquarium",
        "sea otter aquarium tank",
        "otter swimming aquarium",
        "otter exhibit zoo",
        "river otter zoo",
        "otter playing water zoo",
        "otter holding hands aquarium",
        "otter floating water zoo",
        "otter close up zoo",
        "otter face close up",
        "baby otter zoo",
        # Depicted
        "otter stuffed animal toy",
        "otter plush toy",
        "otter in children's book illustration",
        "otter figurine toy",
        "otter bath toy",
        "otter nursery wall art",
        "sea otter toy",
        "otter on baby blanket print",
    ],
}

# Combine all categories
ALL_CATEGORIES = {
    "fridge": FRIDGE_QUERIES,
    "common": COMMON_ANIMAL_QUERIES,
    "zoo": ZOO_ANIMAL_QUERIES,
    "aquarium": AQUARIUM_ANIMAL_QUERIES,
    "other": OTHER_ANIMAL_QUERIES,
}

ALL_QUERIES = {}
for cat_queries in ALL_CATEGORIES.values():
    ALL_QUERIES.update(cat_queries)


def download_bing_images(query: str, max_images: int = 50) -> list[Path]:
    """Download images from Bing Image Search using icrawler."""
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("  pip install icrawler  (required for Bing image search)")
        sys.exit(1)

    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="seeky_animal_"))

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

    tmp_dir = Path(tempfile.mkdtemp(prefix="seeky_animal_"))

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

        img = img.resize(CROP_SIZE, Image.LANCZOS)

        content_hash = hashlib.md5(img.tobytes()[:10000]).hexdigest()[:8]
        fname = f"{word}_animal_{content_hash}.jpg"
        out_path = word_dir / fname

        if out_path.exists():
            continue

        img.save(out_path, "JPEG", quality=90)
        saved += 1

    return saved


def collect_for_word(word: str, queries: list[str], max_per_query: int = 50,
                     engine: str = "bing"):
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

        saved = process_and_save(paths, word, word_dir)
        total_new += saved
        print(f"got {len(paths)}, saved {saved}")

        # Rate limit: DDG needs longer cooldown to avoid 403s
        delay = 5 if engine == "ddg" else 2
        time.sleep(delay)

    new_total = existing + total_new
    print(f"  [{word}] Total: {new_total} images (+{total_new} new)")
    return total_new


def main():
    parser = argparse.ArgumentParser(
        description="Collect phone-camera training images for animals and fridge")
    parser.add_argument("--word", type=str, help="Collect for a specific word")
    parser.add_argument("--category", type=str,
                        choices=list(ALL_CATEGORIES.keys()),
                        help="Collect for one category (fridge/common/zoo/aquarium/other)")
    parser.add_argument("--max-per-query", type=int, default=50,
                        help="Max images per search query (default: 50)")
    parser.add_argument("--engine", choices=["bing", "ddg"], default="bing",
                        help="Search engine (bing or ddg)")
    parser.add_argument("--list", action="store_true", help="List available words")
    args = parser.parse_args()

    if args.list:
        for cat_name, cat_queries in ALL_CATEGORIES.items():
            print(f"\n=== {cat_name.upper()} ===")
            for word, queries in sorted(cat_queries.items()):
                existing = len(list((OUTPUT_DIR / word.replace(" ", "_")).glob("*.jpg")))
                print(f"  {word}: {len(queries)} queries ({existing} existing images)")
        return

    if args.word:
        if args.word not in ALL_QUERIES:
            print(f"No queries for '{args.word}'. Run with --list to see available words.")
            sys.exit(1)
        collect_for_word(args.word, ALL_QUERIES[args.word], args.max_per_query, args.engine)
    elif args.category:
        cat_queries = ALL_CATEGORIES[args.category]
        total = 0
        for word, queries in sorted(cat_queries.items()):
            total += collect_for_word(word, queries, args.max_per_query, args.engine)
        print(f"\n=== DONE: {total} new images for {args.category} category ===")
    else:
        # Collect all categories in priority order
        priority_order = ["fridge", "common", "zoo", "aquarium", "other"]
        grand_total = 0
        for cat_name in priority_order:
            cat_queries = ALL_CATEGORIES[cat_name]
            print(f"\n{'='*60}")
            print(f"  CATEGORY: {cat_name.upper()}")
            print(f"{'='*60}")
            cat_total = 0
            for word, queries in sorted(cat_queries.items()):
                cat_total += collect_for_word(word, queries, args.max_per_query, args.engine)
            grand_total += cat_total
            print(f"\n  [{cat_name}] subtotal: {cat_total} new images")

        print(f"\n{'='*60}")
        print(f"  GRAND TOTAL: {grand_total} new images across all categories")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("  1. Review images: open data/seeky_training/all/<word>/ and remove bad images")
        print("  2. Retrain: python scripts/train_classifier.py")
        print("  3. Convert: python scripts/convert_to_coreml.py")
        print("  4. Test on device")


if __name__ == "__main__":
    main()
