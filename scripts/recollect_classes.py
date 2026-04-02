#!/usr/bin/env python3
"""Re-collect training data for specific classes that need better source mappings.

Clears existing data for the specified classes, re-collects from corrected
sources, and re-splits into train/val.

Usage:
    python scripts/recollect_classes.py
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path

# Reuse collection infrastructure from the main script
sys.path.insert(0, str(Path(__file__).parent))
from collect_training_data import (
    OUTPUT_DIR, TARGET_PER_CLASS, VAL_FRACTION, CROP_SIZE, MIN_CROP_PX,
    collect_from_open_images, collect_from_coco, crop_and_save
)

# Classes to re-collect with corrected source mappings
RECOLLECT = {
    # light: removed Traffic light (wrong context for toddlers), added Lantern + Flashlight
    "light": [("oi7", "Light bulb"), ("oi7", "Lantern"), ("oi7", "Flashlight")],
    # monitor: removed COCO "tv" which overlapped with the tv class
    "monitor": [("oi7", "Computer monitor")],
    # panda: removed Red panda (looks like raccoon, confuses model)
    "panda": [("oi7", "Giant panda")],
    # lamp: sources were fine, but re-collect to ensure no overlap with light
    "lamp": [("oi7", "Lamp"), ("oi7", "Table lamp")],
    # laptop: add more sources to distinguish from monitor
    "laptop": [("oi7", "Laptop"), ("coco", "laptop")],
    # tv: sources are fine but re-collect to ensure clean separation from monitor
    "tv": [("oi7", "Television"), ("coco", "tv")],
}


def clear_class(word: str):
    """Remove existing data for a class from all/, train/, val/."""
    word_slug = word.replace(" ", "_")
    for split in ["all", "train", "val"]:
        d = OUTPUT_DIR / split / word_slug
        if d.exists():
            count = len(list(d.glob("*.jpg")))
            shutil.rmtree(d)
            print(f"  Cleared {split}/{word_slug} ({count} images)")


def collect_for_word(word: str, sources: list) -> int:
    """Collect training images for a single word from given sources."""
    word_slug = word.replace(" ", "_")
    word_dir = OUTPUT_DIR / "all" / word_slug
    word_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    remaining = TARGET_PER_CLASS
    samples_per_source = max(400, (remaining * 4) // len(sources))

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

        dataset.delete()

    return total


def split_class(word: str):
    """Split a single class into train/val."""
    random.seed(42)
    word_slug = word.replace(" ", "_")
    all_dir = OUTPUT_DIR / "all" / word_slug
    train_dir = OUTPUT_DIR / "train" / word_slug
    val_dir = OUTPUT_DIR / "val" / word_slug

    if not all_dir.exists():
        print(f"  [WARN] No all/ directory for '{word}'")
        return

    images = sorted(all_dir.glob("*.jpg"))
    random.shuffle(images)

    val_count = max(1, int(len(images) * VAL_FRACTION))
    val_images = images[:val_count]
    train_images = images[val_count:]

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for img_path in train_images:
        shutil.copy2(img_path, train_dir / img_path.name)
    for img_path in val_images:
        shutil.copy2(img_path, val_dir / img_path.name)

    print(f"  {word}: {len(train_images)} train, {len(val_images)} val")


def main():
    print(f"Re-collecting {len(RECOLLECT)} classes with corrected sources")
    print(f"Output: {OUTPUT_DIR}")
    print()

    for word, sources in RECOLLECT.items():
        print(f"\n{'='*60}")
        print(f"Re-collecting: {word}")
        print(f"Sources: {sources}")
        print(f"{'='*60}")

        # Clear old data
        clear_class(word)

        # Collect new data
        count = collect_for_word(word, sources)
        print(f"  Collected {count} images for '{word}'")

        # Re-split
        split_class(word)

    print(f"\n{'='*60}")
    print("RE-COLLECTION COMPLETE")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. python scripts/curate_training_data.py --execute --re-split")
    print("  2. python scripts/train_classifier.py")
    print("  3. python scripts/convert_to_coreml.py")


if __name__ == "__main__":
    main()
