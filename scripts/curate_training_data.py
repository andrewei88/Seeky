#!/usr/bin/env python3
"""Automated training data curation.

Runs the trained model on ALL training images and quarantines likely-mislabeled
ones (where the model confidently predicts a different class than the folder).

This is a self-consistency check: if the model strongly disagrees with the label,
the image is probably mislabeled or very ambiguous — either way, it hurts training.

Usage:
    python3 scripts/curate_training_data.py              # dry run (report only)
    python3 scripts/curate_training_data.py --execute     # actually move files
    python3 scripts/curate_training_data.py --execute --re-split  # move + re-split train/val
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "arya_training"
ALL_DIR = DATA_DIR / "all"
QUARANTINE_DIR = DATA_DIR / "quarantined"
IMAGE_SIZE = 224

# Backbone must match train_classifier.py
BACKBONE = "fastvit_t12"

# Confidence threshold: if model is THIS confident the image belongs to a
# DIFFERENT class, quarantine it. Set conservatively — we'd rather keep a
# borderline image than remove a valid one.
QUARANTINE_THRESHOLD = 0.70
# Also quarantine if model's confidence FOR the labeled class is below this
LOW_SELF_CONFIDENCE = 0.05


class ARYAClassifier(nn.Module):
    """Must match train_classifier.py exactly."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.class_head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.class_head(features)
        return logits, features


def load_model(num_classes: int):
    model = ARYAClassifier(num_classes)
    weights_path = MODEL_DIR / "arya_classifier_best.pth"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def scan_images(all_dir: Path, classes: list[str]):
    """Scan all training images, return list of (path, class_name, class_idx)."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    images = []
    for cls in classes:
        cls_dir = all_dir / cls
        if not cls_dir.exists():
            print(f"  WARNING: Missing class directory: {cls_dir}")
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                images.append((img_path, cls, class_to_idx[cls]))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Actually move files (default is dry run)")
    parser.add_argument("--re-split", action="store_true",
                        help="Re-split train/val after quarantining")
    parser.add_argument("--threshold", type=float, default=QUARANTINE_THRESHOLD,
                        help=f"Quarantine if model confidence for OTHER class > this (default {QUARANTINE_THRESHOLD})")
    parser.add_argument("--low-self", type=float, default=LOW_SELF_CONFIDENCE,
                        help=f"Quarantine if model confidence for LABELED class < this (default {LOW_SELF_CONFIDENCE})")
    args = parser.parse_args()

    # Load classes
    classes_path = MODEL_DIR / "arya_classes.json"
    with open(classes_path) as f:
        classes = json.load(f)
    num_classes = len(classes)
    print(f"Classes: {num_classes}")

    # Load model
    print("Loading model...")
    model = load_model(num_classes)

    # Transform (same as validation)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Scan images
    print(f"Scanning {ALL_DIR}...")
    images = scan_images(ALL_DIR, classes)
    print(f"Found {len(images)} images across {num_classes} classes\n")

    # Run inference on all images
    to_quarantine = []  # (path, labeled_class, predicted_class, labeled_conf, predicted_conf)
    class_stats = defaultdict(lambda: {"total": 0, "quarantined": 0})
    confusion_counts = defaultdict(lambda: defaultdict(int))

    print("Running inference...")
    with torch.no_grad():
        for i, (img_path, labeled_cls, labeled_idx) in enumerate(images):
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0)
            except Exception as e:
                print(f"  ERROR loading {img_path}: {e}")
                to_quarantine.append((img_path, labeled_cls, "CORRUPT", 0.0, 0.0))
                class_stats[labeled_cls]["total"] += 1
                class_stats[labeled_cls]["quarantined"] += 1
                continue

            logits, _ = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

            labeled_conf = probs[labeled_idx].item()
            pred_idx = probs.argmax().item()
            pred_conf = probs[pred_idx].item()
            pred_cls = classes[pred_idx]

            class_stats[labeled_cls]["total"] += 1

            # Quarantine conditions:
            # 1. Model confidently predicts DIFFERENT class
            # 2. Model has near-zero confidence for the labeled class
            should_quarantine = False
            reason = ""

            if pred_idx != labeled_idx and pred_conf >= args.threshold:
                should_quarantine = True
                reason = f"model says '{pred_cls}'({pred_conf:.3f}), not '{labeled_cls}'({labeled_conf:.3f})"
            elif labeled_conf < args.low_self and pred_idx != labeled_idx:
                should_quarantine = True
                reason = f"very low self-conf({labeled_conf:.3f}), model prefers '{pred_cls}'({pred_conf:.3f})"

            if should_quarantine:
                to_quarantine.append((img_path, labeled_cls, pred_cls, labeled_conf, pred_conf))
                class_stats[labeled_cls]["quarantined"] += 1
                confusion_counts[labeled_cls][pred_cls] += 1

            if (i + 1) % 500 == 0:
                sys.stdout.write(f"\r  Processed {i+1}/{len(images)} — quarantine candidates: {len(to_quarantine)}")
                sys.stdout.flush()

    print(f"\r  Processed {len(images)}/{len(images)} — quarantine candidates: {len(to_quarantine)}")

    # Report
    print(f"\n{'='*70}")
    print(f"CURATION REPORT (threshold={args.threshold}, low_self={args.low_self})")
    print(f"{'='*70}")
    print(f"  Total images scanned: {len(images)}")
    print(f"  Images to quarantine: {len(to_quarantine)} ({100*len(to_quarantine)/len(images):.1f}%)")
    print(f"  Images remaining:     {len(images) - len(to_quarantine)}")

    # Per-class quarantine stats (sorted by % quarantined)
    print(f"\n{'='*70}")
    print("PER-CLASS QUARANTINE (sorted by % removed)")
    print(f"{'='*70}")

    sorted_stats = sorted(
        class_stats.items(),
        key=lambda x: x[1]["quarantined"] / max(x[1]["total"], 1),
        reverse=True,
    )

    for cls, stats in sorted_stats:
        t = stats["total"]
        q = stats["quarantined"]
        if q == 0:
            continue
        pct = 100 * q / t
        confused_with = sorted(confusion_counts[cls].items(), key=lambda x: -x[1])[:3]
        confused_str = ", ".join(f"{c}({n})" for c, n in confused_with)
        print(f"  {cls:15s}: {q:3d}/{t:3d} removed ({pct:4.1f}%) — confused with: {confused_str}")

    unaffected = sum(1 for _, s in class_stats.items() if s["quarantined"] == 0)
    print(f"\n  {unaffected} classes have no quarantine candidates")

    # Show sample quarantined images
    print(f"\n{'='*70}")
    print("SAMPLE QUARANTINED IMAGES (first 30)")
    print(f"{'='*70}")
    for path, labeled, predicted, l_conf, p_conf in to_quarantine[:30]:
        print(f"  {path.name:30s} labeled='{labeled}' → model='{predicted}' (label_conf={l_conf:.3f}, pred_conf={p_conf:.3f})")

    if not args.execute:
        print(f"\n{'='*70}")
        print("DRY RUN — no files moved. Run with --execute to quarantine files.")
        print(f"{'='*70}")

        # Save report for reference
        report = {
            "threshold": args.threshold,
            "low_self": args.low_self,
            "total_images": len(images),
            "quarantine_count": len(to_quarantine),
            "quarantine_files": [
                {
                    "path": str(p),
                    "labeled": l,
                    "predicted": pred,
                    "label_conf": lc,
                    "pred_conf": pc,
                }
                for p, l, pred, lc, pc in to_quarantine
            ],
            "per_class": {
                cls: {"total": s["total"], "quarantined": s["quarantined"]}
                for cls, s in class_stats.items()
            },
        }
        report_path = MODEL_DIR / "curation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {report_path}")
        return

    # Execute quarantine
    print(f"\nMoving {len(to_quarantine)} files to {QUARANTINE_DIR}...")
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    for path, labeled_cls, predicted_cls, _, _ in to_quarantine:
        dest_dir = QUARANTINE_DIR / labeled_cls
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / path.name
        if dest.exists():
            dest = dest_dir / f"{path.stem}_dup{path.suffix}"
        shutil.move(str(path), str(dest))

    print(f"  Moved {len(to_quarantine)} files to quarantine")

    # Re-split if requested
    if args.re_split:
        print("\nRe-splitting train/val...")
        resplit(classes)


def resplit(classes: list[str], val_ratio: float = 0.15):
    """Re-create train/val split from the (now cleaned) all/ directory."""
    import random
    random.seed(42)

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    # Clear existing splits
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)

    total_train = 0
    total_val = 0

    for cls in classes:
        src_dir = ALL_DIR / cls
        if not src_dir.exists():
            continue

        imgs = sorted([
            p for p in src_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ])
        random.shuffle(imgs)

        n_val = max(1, int(len(imgs) * val_ratio))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)

        for p in train_imgs:
            shutil.copy2(str(p), str(train_dir / cls / p.name))
        for p in val_imgs:
            shutil.copy2(str(p), str(val_dir / cls / p.name))

        total_train += len(train_imgs)
        total_val += len(val_imgs)

    print(f"  Train: {total_train} images")
    print(f"  Val:   {total_val} images")
    print("  Done!")


if __name__ == "__main__":
    main()
