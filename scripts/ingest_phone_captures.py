#!/usr/bin/env python3
"""Ingest training captures from the iOS app into the training pipeline.

When users correct misidentified objects in the app, cropped images are saved
to the app's documents directory. This script copies those images into the
training data, re-splits train/val, and optionally retrains the model.

Usage:
    # From exported zip (via share sheet in the app):
    python scripts/ingest_phone_captures.py path/to/seeky_training_captures.zip

    # From a folder already placed at data/phone_captures/:
    python scripts/ingest_phone_captures.py

    # To also retrain and convert in one step:
    python scripts/ingest_phone_captures.py --retrain [path/to/zip]
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
CAPTURES_DIR = PROJECT_ROOT / "data" / "phone_captures"
TRAINING_DIR = PROJECT_ROOT / "data" / "seeky_training"
CROP_SIZE = (256, 256)
VAL_FRACTION = 0.15


def ingest_captures():
    """Copy phone captures into the training data all/ directory."""
    if not CAPTURES_DIR.exists():
        print(f"No captures directory found at: {CAPTURES_DIR}")
        print("Export training_captures/ from the iOS app and place it at data/phone_captures/")
        return False

    class_dirs = [d for d in CAPTURES_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"No class folders found in {CAPTURES_DIR}")
        return False

    total_ingested = 0
    classes_updated = []

    for class_dir in sorted(class_dirs):
        word = class_dir.name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        if not images:
            continue

        all_dir = TRAINING_DIR / "all" / word
        all_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(all_dir.glob("*.jpg")))
        count = 0

        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(CROP_SIZE, Image.LANCZOS)
                new_name = f"{word}_phone_{existing + count:05d}.jpg"
                img.save(all_dir / new_name, "JPEG", quality=95)
                count += 1
            except Exception as e:
                print(f"  [WARN] Failed to process {img_path.name}: {e}")

        if count > 0:
            total_ingested += count
            classes_updated.append(word)
            print(f"  {word}: ingested {count} phone captures (total in all/: {existing + count})")

    if total_ingested == 0:
        print("No images found to ingest.")
        return False

    print(f"\nIngested {total_ingested} images across {len(classes_updated)} classes")
    return True


def resplit_classes(classes: list[str] | None = None):
    """Re-split specified classes (or all) into train/val."""
    random.seed(42)

    if classes is None:
        classes = [d.name for d in (TRAINING_DIR / "all").iterdir() if d.is_dir()]

    for word in sorted(classes):
        all_dir = TRAINING_DIR / "all" / word
        if not all_dir.exists():
            continue

        train_dir = TRAINING_DIR / "train" / word
        val_dir = TRAINING_DIR / "val" / word

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

        print(f"  {word}: {len(train_images)} train, {len(val_images)} val")


def extract_zip(zip_path: Path):
    """Extract a zip file exported from the app into the captures directory."""
    print(f"Extracting zip: {zip_path}")
    if not zip_path.exists():
        print(f"Zip file not found: {zip_path}")
        return False

    if CAPTURES_DIR.exists():
        shutil.rmtree(CAPTURES_DIR)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CAPTURES_DIR)

    # The zip may contain a top-level directory (training_captures/). Flatten if needed.
    subdirs = [d for d in CAPTURES_DIR.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and not list(CAPTURES_DIR.glob("*.jpg")):
        nested = subdirs[0]
        for item in nested.iterdir():
            shutil.move(str(item), str(CAPTURES_DIR / item.name))
        nested.rmdir()

    print(f"Extracted to: {CAPTURES_DIR}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest phone captures into training pipeline")
    parser.add_argument("zip_path", nargs="?", help="Path to exported zip file (optional)")
    parser.add_argument("--retrain", action="store_true", help="Also retrain and convert the model")
    args = parser.parse_args()

    if args.zip_path:
        if not extract_zip(Path(args.zip_path)):
            sys.exit(1)

    print("Ingesting phone captures into training pipeline")
    print(f"Captures: {CAPTURES_DIR}")
    print(f"Training: {TRAINING_DIR}")
    print()

    if not ingest_captures():
        sys.exit(1)

    # Find which classes were updated
    classes_with_phone = []
    for d in (TRAINING_DIR / "all").iterdir():
        if d.is_dir() and list(d.glob("*_phone_*.jpg")):
            classes_with_phone.append(d.name)

    print(f"\nRe-splitting {len(classes_with_phone)} updated classes...")
    resplit_classes(classes_with_phone)

    if args.retrain:
        print("\n" + "=" * 60)
        print("RETRAINING MODEL")
        print("=" * 60)
        subprocess.run([sys.executable, "scripts/train_classifier.py"], cwd=str(PROJECT_ROOT), check=True)

        print("\n" + "=" * 60)
        print("CONVERTING TO COREML")
        print("=" * 60)
        subprocess.run([sys.executable, "scripts/convert_to_coreml.py"], cwd=str(PROJECT_ROOT), check=True)

        print("\nDone! New model is at Seeky/Resources/SeekyClassifier.mlpackage")
        print("Build and deploy to device to test.")
    else:
        print("\nCaptures ingested. To retrain:")
        print("  python scripts/train_classifier.py")
        print("  python scripts/convert_to_coreml.py")


if __name__ == "__main__":
    main()
