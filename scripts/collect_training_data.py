#!/usr/bin/env python3
"""Collect training images for the 107-word ARYA vocabulary.

Uses FiftyOne to pull cropped object images from Open Images V7 and COCO,
organized into train/val splits for MobileNetV3 fine-tuning.

Usage:
    pip install fiftyone Pillow
    python scripts/collect_training_data.py

Output: data/arya_training/{train,val}/{word}/  (~300-500 images per class)
"""

import json
import os
import shutil
import sys
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
VOCAB_PATH = PROJECT_ROOT / "ARYA" / "Resources" / "vocabulary.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "arya_training"
TARGET_PER_CLASS = 500       # aim for this many images per class
MIN_PER_CLASS = 200          # warn if we get fewer than this
VAL_FRACTION = 0.15          # 15% val split
CROP_SIZE = (256, 256)       # resize crops to this (slightly larger than 224 for augmentation margin)
MIN_CROP_PX = 20             # skip tiny bounding boxes (< 20px on either side)

# ── Word → Dataset Label Mapping ────────────────────────────────────────────
# Maps each ARYA vocab word to a list of (dataset, label) tuples to query.
# "oi7" = Open Images V7 detection labels
# "coco" = COCO 2017 detection labels
# Labels are case-sensitive and must match the dataset's label vocabulary exactly.
#
# Words with no good dataset match are in MANUAL_WORDS — user must supply images.
WORD_TO_SOURCES = {
    "apple":      [("oi7", "Apple"), ("coco", "apple")],
    "bag":        [("oi7", "Handbag"), ("oi7", "Backpack"), ("coco", "handbag"), ("coco", "backpack")],
    "ball":       [("oi7", "Ball"), ("oi7", "Tennis ball"), ("oi7", "Football")],
    "banana":     [("oi7", "Banana"), ("coco", "banana")],
    "bathtub":    [("oi7", "Bathtub")],
    "bear":       [("oi7", "Bear"), ("coco", "bear")],
    "bed":        [("oi7", "Bed"), ("coco", "bed")],
    "bench":      [("oi7", "Bench"), ("coco", "bench")],
    "bird":       [("oi7", "Bird"), ("coco", "bird")],
    "blanket":    [("oi7", "Blanket")],
    "block":      [],  # manual — toy building blocks not in standard datasets
    "book":       [("oi7", "Book"), ("coco", "book")],
    "bottle":     [("oi7", "Bottle"), ("coco", "bottle")],
    "bowl":       [("oi7", "Bowl"), ("coco", "bowl")],
    "box":        [("oi7", "Box")],
    "bread":      [("oi7", "Bread")],
    "bus":        [("oi7", "Bus"), ("coco", "bus")],
    "butterfly":  [("oi7", "Butterfly")],
    "cake":       [("oi7", "Cake"), ("coco", "cake")],
    "can":        [("oi7", "Tin can"), ("oi7", "Drink")],
    "car":        [("oi7", "Car"), ("coco", "car")],
    "cat":        [("oi7", "Cat"), ("coco", "cat")],
    "chair":      [("oi7", "Chair"), ("oi7", "Office chair"), ("oi7", "Rocking chair"), ("coco", "chair")],
    "cheese":     [("oi7", "Cheese")],
    "chicken":    [("oi7", "Chicken")],
    "clock":      [("oi7", "Clock"), ("coco", "clock")],
    "cloud":      [],  # manual — clouds are scene-level, not object detections
    "cookie":     [("oi7", "Cookie")],
    "couch":      [("oi7", "Couch"), ("coco", "couch")],
    "cow":        [("oi7", "Cattle"), ("coco", "cow")],
    "crayon":     [],  # manual — not in standard detection datasets
    "cup":        [("oi7", "Coffee cup"), ("oi7", "Cup"), ("coco", "cup")],
    "dog":        [("oi7", "Dog"), ("coco", "dog")],
    "doll":       [("oi7", "Doll")],
    "door":       [("oi7", "Door"), ("oi7", "French door")],
    "duck":       [("oi7", "Duck")],
    "ear":        [("oi7", "Human ear")],
    "egg":        [("oi7", "Egg")],
    "elephant":   [("oi7", "Elephant"), ("coco", "elephant")],
    "eye":        [("oi7", "Human eye")],
    "face":       [("oi7", "Human face")],
    "fan":        [("oi7", "Ceiling fan")],
    "fence":      [("oi7", "Fence")],
    "fish":       [("oi7", "Fish"), ("oi7", "Goldfish")],
    "flower":     [("oi7", "Flower"), ("oi7", "Rose")],
    "foot":       [("oi7", "Human foot")],
    "fork":       [("oi7", "Fork"), ("coco", "fork")],
    "frog":       [("oi7", "Frog")],
    "glass":      [("oi7", "Wine glass"), ("coco", "wine glass")],
    "glasses":    [("oi7", "Glasses"), ("oi7", "Sunglasses")],
    "grass":      [],  # manual — scene/texture, not an object detection
    "hand":       [("oi7", "Human hand")],
    "hat":        [("oi7", "Hat"), ("oi7", "Fedora"), ("oi7", "Cowboy hat")],
    "horse":      [("oi7", "Horse"), ("coco", "horse")],
    "jacket":     [("oi7", "Jacket")],
    "key":        [],  # manual — keys too small/rare in detection datasets
    "keyboard":   [("oi7", "Computer keyboard"), ("coco", "keyboard")],
    "knife":      [("oi7", "Knife"), ("coco", "knife")],
    "lamp":       [("oi7", "Lamp"), ("oi7", "Table lamp"), ("oi7", "Desk lamp"), ("oi7", "Floor lamp")],
    "laptop":     [("oi7", "Laptop"), ("coco", "laptop")],
    "leaf":       [],  # manual — leaves are too generic in detection datasets
    "light":      [("oi7", "Light bulb"), ("oi7", "Lantern"), ("oi7", "Flashlight"), ("oi7", "Chandelier"), ("oi7", "Light switch")],
    "lion":       [("oi7", "Lion")],
    "mirror":     [("oi7", "Mirror")],
    "monkey":     [("oi7", "Monkey")],
    "moon":       [("oi7", "Moon")],
    "nose":       [("oi7", "Human nose")],
    "orange":     [("oi7", "Orange"), ("coco", "orange")],
    "pan":        [("oi7", "Frying pan"), ("oi7", "Wok")],
    "pants":      [("oi7", "Jeans"), ("oi7", "Shorts")],
    "paper":      [],  # manual — paper is too generic
    "pen":        [("oi7", "Pen")],
    "pencil":     [("oi7", "Pencil")],
    "phone":      [("oi7", "Mobile phone"), ("coco", "cell phone")],
    "picture":    [("oi7", "Picture frame")],
    "pig":        [("oi7", "Pig")],
    "pillow":     [("oi7", "Pillow")],
    "pizza":      [("oi7", "Pizza"), ("coco", "pizza")],
    "plate":      [("oi7", "Plate")],
    "pot":        [("oi7", "Flowerpot"), ("oi7", "Cooking pot"), ("coco", "potted plant")],
    "rabbit":     [("oi7", "Rabbit")],
    "rain":       [],  # manual — rain is a weather condition, not a detectable object
    "remote":     [("coco", "remote")],
    "rock":       [],  # manual — rocks not well-covered in detection datasets
    "scissors":   [("oi7", "Scissors"), ("coco", "scissors")],
    "shelf":      [("oi7", "Shelf")],
    "shirt":      [("oi7", "Shirt")],
    "shoe":       [("oi7", "Shoe"), ("oi7", "High heels"), ("oi7", "Boot")],
    "sink":       [("oi7", "Sink"), ("coco", "sink")],
    "soap":       [],  # manual — soap not in standard detection datasets
    "sock":       [("oi7", "Sock")],
    "speaker":    [("oi7", "Loudspeaker")],
    "spoon":      [("oi7", "Spoon"), ("coco", "spoon")],
    "star":       [],  # manual — stars are celestial, not object detections
    "sun":        [],  # manual — same as star
    "table":      [("oi7", "Table"), ("oi7", "Coffee table"), ("oi7", "Kitchen & dining room table"), ("coco", "dining table")],
    "teddy bear": [("oi7", "Teddy bear"), ("coco", "teddy bear")],
    "toilet":     [("oi7", "Toilet"), ("coco", "toilet")],
    "toothbrush": [("oi7", "Toothbrush"), ("coco", "toothbrush")],
    "towel":      [("oi7", "Towel")],
    "tree":       [("oi7", "Tree")],
    "truck":      [("oi7", "Truck"), ("coco", "truck")],
    "turtle":     [("oi7", "Turtle"), ("oi7", "Tortoise")],
    "tv":         [("oi7", "Television"), ("oi7", "Plasma tv"), ("coco", "tv")],
    "umbrella":   [("oi7", "Umbrella"), ("coco", "umbrella")],
    "window":     [("oi7", "Window")],
    "monitor":    [("oi7", "Computer monitor")],
    # ── New classes (v2) ────────────────────────────────────────────────────
    "blackberry":  [],  # manual — not in standard detection datasets
    "blueberry":   [],  # manual — not in standard detection datasets
    "deer":        [("oi7", "Deer")],
    "giraffe":     [("oi7", "Giraffe"), ("coco", "giraffe")],
    "grape":       [("oi7", "Grape")],
    "kiwi":        [],  # manual — kiwi fruit not reliably in detection datasets
    "lemon":       [("oi7", "Lemon")],
    "mango":       [("oi7", "Mango")],
    "panda":       [("oi7", "Panda")],
    "penguin":     [("oi7", "Penguin")],
    "pineapple":   [("oi7", "Pineapple")],
    "raspberry":   [],  # manual — not in standard detection datasets
    "sheep":       [("oi7", "Sheep"), ("coco", "sheep")],
    "snake":       [("oi7", "Snake"), ("oi7", "Cobra")],
    "stairs":      [("oi7", "Stairs")],
    "strawberry":  [("oi7", "Strawberry")],
    "tiger":       [("oi7", "Tiger")],
    "toilet paper": [("oi7", "Toilet paper")],
    "watermelon":  [("oi7", "Watermelon")],
    "zebra":       [("oi7", "Zebra"), ("coco", "zebra")],
    # ── New classes (v3) — sea animals, land animals, fruits, plants ───────
    "avocado":     [],  # manual — not in standard detection datasets
    "cactus":      [],  # manual — cacti not reliably in OI7
    "camel":       [("oi7", "Camel")],
    "cherry":      [("oi7", "Cherry")],
    "coconut":     [("oi7", "Coconut")],
    "crab":        [("oi7", "Crab")],
    "cupboard":    [("oi7", "Cupboard"), ("oi7", "Cabinetry")],
    "dishwasher":  [("oi7", "Dishwasher")],
    "dolphin":     [("oi7", "Dolphin")],
    "fox":         [("oi7", "Fox"), ("oi7", "Red fox")],
    "fridge":      [("oi7", "Refrigerator")],
    "goat":        [("oi7", "Goat")],
    "hippo":       [("oi7", "Hippopotamus")],
    "jellyfish":   [("oi7", "Jellyfish")],
    "microwave":   [("oi7", "Microwave oven")],
    "mushroom":    [("oi7", "Mushroom")],
    "octopus":     [("oi7", "Octopus")],
    "oven":        [("oi7", "Oven")],
    "owl":         [("oi7", "Owl")],
    "parrot":      [("oi7", "Parrot")],
    "peach":       [("oi7", "Peach")],
    "pear":        [("oi7", "Pear")],
    "seal":        [("oi7", "Seal")],
    "shark":       [("oi7", "Shark"), ("oi7", "Great white shark")],
    "squirrel":    [("oi7", "Squirrel")],
    "sunflower":   [("oi7", "Sunflower")],
    "toaster":     [("oi7", "Toaster")],
    "whale":       [("oi7", "Whale"), ("oi7", "Blue whale")],
}

# Words that need manual image collection
MANUAL_WORDS = [w for w, sources in WORD_TO_SOURCES.items() if not sources]


def load_vocabulary():
    """Load vocabulary words from vocabulary.json."""
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    return [entry["word"] for entry in vocab]


def collect_from_open_images(label: str, max_samples: int) -> fo.Dataset:
    """Download Open Images V7 samples with the given detection label."""
    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=[label],
            max_samples=max_samples,
            shuffle=True,
            dataset_name=None,  # auto-generate unique name
        )
        return dataset
    except Exception as e:
        print(f"  [WARN] Failed to load OI7 '{label}': {e}")
        return None


def collect_from_coco(label: str, max_samples: int) -> fo.Dataset:
    """Download COCO 2017 samples with the given detection label."""
    try:
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            label_types=["detections"],
            classes=[label],
            max_samples=max_samples,
            shuffle=True,
            dataset_name=None,
        )
        return dataset
    except Exception as e:
        print(f"  [WARN] Failed to load COCO '{label}': {e}")
        return None


def crop_and_save(dataset: fo.Dataset, label: str, word: str, output_dir: Path,
                  existing_count: int, max_total: int) -> int:
    """Crop bounding boxes for `label` from dataset samples, save as images.

    Returns the number of images saved.
    """
    saved = 0
    for sample in dataset:
        if existing_count + saved >= max_total:
            break

        if sample.filepath is None or not os.path.exists(sample.filepath):
            continue

        # Get detections matching our label
        detections = sample.ground_truth.detections if sample.ground_truth else []
        matching = [d for d in detections if d.label == label]

        if not matching:
            continue

        try:
            img = Image.open(sample.filepath).convert("RGB")
        except Exception:
            continue

        w, h = img.size

        for det in matching:
            if existing_count + saved >= max_total:
                break

            # FiftyOne bounding boxes are [x, y, width, height] in relative coords
            bx, by, bw, bh = det.bounding_box
            x1 = int(bx * w)
            y1 = int(by * h)
            x2 = int((bx + bw) * w)
            y2 = int((by + bh) * h)

            # Skip tiny crops
            if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                continue

            # Add 10% padding
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            crop = img.crop((x1, y1, x2, y2))
            crop = crop.resize(CROP_SIZE, Image.LANCZOS)

            fname = f"{word}_{existing_count + saved:05d}.jpg"
            crop.save(output_dir / fname, "JPEG", quality=90)
            saved += 1

    return saved


def collect_for_word(word: str):
    """Collect training images for a single vocabulary word."""
    sources = WORD_TO_SOURCES.get(word, [])
    if not sources:
        return 0

    word_dir = OUTPUT_DIR / "all" / word.replace(" ", "_")
    word_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(word_dir.glob("*.jpg")))
    if existing >= TARGET_PER_CLASS:
        print(f"  [{word}] Already have {existing} images, skipping")
        return existing

    total = existing
    remaining = TARGET_PER_CLASS - total

    # Calculate how many samples to request per source
    # Request 4x what we need since not every sample will have a usable crop
    # (some detections are too small, some images have no matching labels)
    samples_per_source = max(300, (remaining * 4) // len(sources))

    for dataset_name, label in sources:
        if total >= TARGET_PER_CLASS:
            break

        print(f"  [{word}] Fetching '{label}' from {dataset_name} (up to {samples_per_source} samples)...")

        if dataset_name == "oi7":
            dataset = collect_from_open_images(label, samples_per_source)
        elif dataset_name == "coco":
            dataset = collect_from_coco(label, samples_per_source)
        else:
            continue

        if dataset is None:
            continue

        saved = crop_and_save(dataset, label, word, word_dir, total, TARGET_PER_CLASS)
        total += saved
        print(f"  [{word}] Got {saved} crops from {dataset_name}:'{label}' (total: {total})")
        sys.stdout.flush()

        # Clean up the temporary dataset
        dataset.delete()

    return total


def split_train_val():
    """Split collected images into train/val directories."""
    import random
    random.seed(42)

    all_dir = OUTPUT_DIR / "all"
    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"

    for word_dir in sorted(all_dir.iterdir()):
        if not word_dir.is_dir():
            continue

        word = word_dir.name
        images = sorted(word_dir.glob("*.jpg"))
        random.shuffle(images)

        val_count = max(1, int(len(images) * VAL_FRACTION))
        val_images = images[:val_count]
        train_images = images[val_count:]

        # Create directories and copy
        (train_dir / word).mkdir(parents=True, exist_ok=True)
        (val_dir / word).mkdir(parents=True, exist_ok=True)

        for img_path in train_images:
            shutil.copy2(img_path, train_dir / word / img_path.name)
        for img_path in val_images:
            shutil.copy2(img_path, val_dir / word / img_path.name)

        print(f"  {word}: {len(train_images)} train, {len(val_images)} val")


def main():
    words = load_vocabulary()
    print(f"Vocabulary: {len(words)} words")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Report manual words upfront
    if MANUAL_WORDS:
        print(f"⚠ {len(MANUAL_WORDS)} words need MANUAL image collection:")
        for w in MANUAL_WORDS:
            print(f"    - {w}")
        print(f"  Place images in: {OUTPUT_DIR}/all/{{word}}/*.jpg")
        print()

    # Collect from datasets
    stats = {}
    for i, word in enumerate(words, 1):
        print(f"[{i}/{len(words)}] Collecting: {word}")
        count = collect_for_word(word)
        stats[word] = count

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)

    low_coverage = []
    zero_coverage = []
    for word, count in sorted(stats.items()):
        status = "OK" if count >= MIN_PER_CLASS else ("MANUAL" if count == 0 else "LOW")
        print(f"  {word:20s}: {count:4d} images  [{status}]")
        if count == 0:
            zero_coverage.append(word)
        elif count < MIN_PER_CLASS:
            low_coverage.append((word, count))

    print(f"\nTotal words: {len(stats)}")
    print(f"Good coverage (>={MIN_PER_CLASS}): {sum(1 for c in stats.values() if c >= MIN_PER_CLASS)}")
    if low_coverage:
        print(f"Low coverage: {len(low_coverage)} — {[f'{w}({c})' for w, c in low_coverage]}")
    if zero_coverage:
        print(f"No coverage (manual needed): {len(zero_coverage)} — {zero_coverage}")

    # Split into train/val
    print("\nSplitting into train/val...")
    split_train_val()

    print("\nDone! Next steps:")
    print(f"  1. Add manual images to {OUTPUT_DIR}/all/{{word}}/ for words with no coverage")
    print(f"  2. Run this script again to re-split after adding manual images")
    print(f"  3. Review images for quality: open {OUTPUT_DIR}/train/ and spot-check")
    print(f"  4. Run scripts/train_classifier.py to train the model")


if __name__ == "__main__":
    main()
