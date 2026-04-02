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
OUTPUT_DIR = PROJECT_ROOT / "data" / "seeky_training"
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
        # Round 4: white/blank screens, calculator apps, document viewers — NOT paper
        "computer monitor showing white screen",
        "monitor with blank white page open",
        "monitor displaying white document",
        "computer monitor with notepad open",
        "monitor showing calculator app",
        "desktop monitor with spreadsheet white background",
        "monitor with white webpage loaded",
        "computer monitor showing blank Word document",
        "monitor displaying PDF white page",
        "iMac showing white background screen",
        "monitor with Google Docs open blank",
        "computer monitor white screen bright room",
        "monitor showing settings menu white",
        "desktop monitor with blank text editor",
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
        # Round 4: white/blank screens, calculator apps, app grids — NOT paper/phone
        "laptop with white screen open on desk",
        "MacBook showing blank white page",
        "laptop displaying white document on screen",
        "laptop with calculator app open",
        "laptop showing notepad blank screen",
        "MacBook with white webpage on screen",
        "laptop with blank Word document open",
        "laptop screen showing white PDF",
        "laptop with settings menu open white",
        "MacBook showing spreadsheet white cells",
        "laptop with Google search page open",
        "laptop displaying blank text editor",
        "laptop screen white background with taskbar",
        "laptop with app icons on screen",
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
    # Round 4: new kitchen/household classes
    "toaster": [
        "toaster on kitchen counter",
        "toaster close up",
        "toaster with bread",
        "silver toaster on counter",
        "toaster front view kitchen",
        "small toaster on table",
        "two slice toaster",
        "toaster oven counter",
    ],
    "oven": [
        "oven in kitchen",
        "oven door close up",
        "kitchen oven front view",
        "open oven with food",
        "oven control panel",
        "wall oven built in kitchen",
        "oven door closed kitchen",
        "stainless steel oven",
    ],
    "dishwasher": [
        "dishwasher in kitchen",
        "dishwasher door open",
        "dishwasher front view kitchen",
        "dishwasher control panel",
        "built in dishwasher kitchen",
        "dishwasher with dishes inside",
        "stainless steel dishwasher",
        "dishwasher closed door kitchen",
    ],
    "cupboard": [
        "kitchen cupboard",
        "cupboard doors close up",
        "open cupboard with dishes",
        "kitchen cabinet close up",
        "wooden cupboard",
        "cupboard in kitchen white",
        "cupboard shelves with plates",
        "kitchen cupboard handle",
    ],
    "fridge": [
        "fridge in kitchen",
        "refrigerator front view",
        "open fridge with food",
        "fridge door close up",
        "stainless steel fridge kitchen",
        "fridge with magnets",
        "kitchen refrigerator",
        "small fridge",
    ],
    "microwave": [
        "microwave on counter",
        "microwave oven kitchen",
        "microwave close up front",
        "microwave door open",
        "microwave on kitchen counter",
        "small microwave",
        "microwave with food inside",
        "countertop microwave",
    ],
    # ── v3: new animals, fruits, plants ────────────────────────────────────
    "whale": [
        "whale toy figurine",
        "whale stuffed animal toy",
        "whale picture in book",
        "blue whale illustration",
        "whale plush toy close up",
        "whale in ocean photo",
        "humpback whale jumping",
        "whale cartoon on poster",
    ],
    "dolphin": [
        "dolphin toy figurine",
        "dolphin stuffed animal",
        "dolphin jumping out of water",
        "dolphin in ocean photo",
        "dolphin plush toy",
        "dolphin picture book page",
        "bottlenose dolphin close up",
    ],
    "shark": [
        "shark toy figurine",
        "shark stuffed animal toy",
        "shark in ocean photo",
        "great white shark photo",
        "shark picture in book",
        "shark toy for kids",
        "baby shark toy",
        "shark plush close up",
    ],
    "crab": [
        "crab on beach photo",
        "crab close up",
        "hermit crab photo",
        "red crab close up",
        "crab toy figurine",
        "crab toy for kids",
        "crab picture in book",
    ],
    "octopus": [
        "octopus stuffed animal toy",
        "octopus plush toy",
        "octopus in ocean photo",
        "octopus close up",
        "octopus toy figurine",
        "octopus picture in book",
        "cute octopus toy",
    ],
    "jellyfish": [
        "jellyfish in aquarium",
        "jellyfish glowing",
        "jellyfish close up photo",
        "jellyfish toy for kids",
        "jellyfish picture in book",
        "jellyfish in ocean",
    ],
    "seal": [
        "seal animal photo",
        "seal on beach",
        "baby seal close up",
        "seal stuffed animal toy",
        "seal plush toy",
        "sea lion photo",
        "harbor seal close up",
        "seal picture in book",
    ],
    "squirrel": [
        "squirrel in yard",
        "squirrel eating nut",
        "squirrel close up photo",
        "squirrel on tree",
        "squirrel in park",
        "grey squirrel photo",
        "squirrel toy figurine",
    ],
    "fox": [
        "fox animal photo",
        "red fox photo",
        "fox close up face",
        "fox in forest",
        "fox toy stuffed animal",
        "fox plush toy",
        "fox picture in book",
    ],
    "owl": [
        "owl close up photo",
        "owl perched on branch",
        "barn owl photo",
        "snowy owl photo",
        "owl stuffed animal toy",
        "owl figurine",
        "owl picture in book",
    ],
    "parrot": [
        "parrot close up photo",
        "parrot on perch",
        "macaw parrot colorful",
        "parrot toy figurine",
        "green parrot photo",
        "parrot plush toy",
        "parrot picture in book",
    ],
    "goat": [
        "goat photo close up",
        "goat on farm",
        "baby goat kid",
        "goat face close up",
        "mountain goat photo",
        "goat toy figurine",
        "goat picture in book",
    ],
    "camel": [
        "camel photo",
        "camel in desert",
        "camel close up face",
        "camel toy figurine",
        "camel stuffed animal",
        "camel picture in book",
        "camel at zoo",
    ],
    "hippo": [
        "hippo in water",
        "hippopotamus photo",
        "hippo close up face",
        "hippo toy figurine",
        "hippo stuffed animal toy",
        "hippo plush toy",
        "hippo picture in book",
        "baby hippo photo",
    ],
    "avocado": [
        "avocado on cutting board",
        "avocado half with pit",
        "avocado on table",
        "avocado close up photo",
        "whole avocado green",
        "sliced avocado on plate",
        "avocado in kitchen",
        "avocado ripe close up",
    ],
    "cactus": [
        "cactus plant in pot",
        "small cactus on desk",
        "cactus houseplant",
        "succulent cactus close up",
        "cactus on windowsill",
        "prickly cactus close up",
        "mini cactus in pot",
        "cactus plant indoor",
    ],
    "cherry": [
        "cherries on table",
        "cherry fruit close up",
        "red cherries in bowl",
        "cherry on stem",
        "fresh cherries photo",
        "cherry fruit plate",
    ],
    "coconut": [
        "coconut on table",
        "coconut half open",
        "coconut close up photo",
        "whole coconut brown",
        "coconut with straw",
        "coconut fruit photo",
    ],
    "peach": [
        "peach fruit on table",
        "peach close up photo",
        "peaches in bowl",
        "ripe peach fruit",
        "peach on cutting board",
        "fresh peach photo",
    ],
    "pear": [
        "pear fruit on table",
        "pear close up photo",
        "green pear fruit",
        "pear in bowl",
        "pear on cutting board",
        "fresh pear fruit photo",
    ],
    "mushroom": [
        "mushroom close up photo",
        "mushroom on cutting board",
        "mushrooms in kitchen",
        "button mushroom close up",
        "mushroom on table",
        "wild mushroom photo",
        "sliced mushrooms",
    ],
    "sunflower": [
        "sunflower close up",
        "sunflower in garden",
        "sunflower bouquet in vase",
        "sunflower field photo",
        "sunflower face close up",
        "single sunflower photo",
    ],
    # ── v3: household item diversity ───────────────────────────────────────
    "chair": [
        "recliner chair in living room",
        "La-Z-Boy recliner",
        "office chair at desk",
        "wooden dining chair",
        "lounge chair close up",
        "bean bag chair",
        "folding chair",
        "rocking chair on porch",
        "accent chair living room",
        "bar stool at counter",
        "upholstered armchair",
    ],
    "table": [
        "glass dining table",
        "glass top coffee table",
        "wooden kitchen table",
        "side table next to couch",
        "end table with lamp",
        "outdoor patio table",
        "round dining table",
        "wood desk table",
        "folding table",
    ],
    "door": [
        "front door of house",
        "closet door closed",
        "sliding glass door",
        "bedroom door open",
        "screen door",
        "garage door from inside",
        "wooden door close up",
        "interior door white",
    ],
    "tv": [
        "flat screen TV on wall",
        "TV mounted on wall",
        "large TV in living room",
        "TV on stand",
        "smart TV close up",
        "TV screen showing movie",
        "wall mounted flat screen TV",
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
