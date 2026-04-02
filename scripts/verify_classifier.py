#!/usr/bin/env python3
"""Verify classifier accuracy on the validation set.

Runs the trained PyTorch model on all val images, reports:
  - Overall top-1 and top-5 accuracy
  - Per-class accuracy (sorted worst to best)
  - Top confusion pairs (what gets confused with what)
  - Classes below accuracy thresholds

Usage:
    python scripts/verify_classifier.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
VAL_DIR = PROJECT_ROOT / "data" / "seeky_training" / "val"
IMAGE_SIZE = 224
BACKBONE = "fastvit_t12"


class SeekyClassifier(nn.Module):
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
    model = SeekyClassifier(num_classes)
    weights_path = MODEL_DIR / "seeky_classifier_best.pth"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    # Load classes
    classes_path = MODEL_DIR / "seeky_classes.json"
    with open(classes_path) as f:
        classes = json.load(f)
    num_classes = len(classes)
    print(f"Classes: {num_classes}")

    # Load model
    print("Loading model...")
    model = load_model(num_classes)

    # Validation transforms (must match training)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load validation dataset
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    # Verify class ordering matches
    folder_classes = val_dataset.classes
    if folder_classes != classes:
        print(f"WARNING: Folder classes don't match seeky_classes.json!")
        print(f"  Folder: {folder_classes[:5]}...")
        print(f"  JSON:   {classes[:5]}...")

    print(f"Validation images: {len(val_dataset)}")
    print(f"Running inference...\n")

    # Run inference
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    # Per-class tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    # confusion[true_class][predicted_class] = count
    confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for images, labels in val_loader:
            logits, _ = model(images)
            probs = torch.softmax(logits, dim=1)

            # Top-1
            _, pred_top1 = probs.max(dim=1)
            correct_top1 += (pred_top1 == labels).sum().item()

            # Top-5
            _, pred_top5 = probs.topk(5, dim=1)
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                pred_label = pred_top1[i].item()
                true_class = classes[true_label]
                pred_class = classes[pred_label]

                class_total[true_class] += 1
                if true_label == pred_label:
                    class_correct[true_class] += 1
                else:
                    confusion[true_class][pred_class] += 1

                if true_label in pred_top5[i]:
                    correct_top5 += 1

            total += labels.size(0)
            sys.stdout.write(f"\r  Processed {total}/{len(val_dataset)}")
            sys.stdout.flush()

    print(f"\n\n{'='*70}")
    print("OVERALL ACCURACY")
    print(f"{'='*70}")
    print(f"  Top-1: {correct_top1}/{total} = {100*correct_top1/total:.1f}%")
    print(f"  Top-5: {correct_top5}/{total} = {100*correct_top5/total:.1f}%")

    # Per-class accuracy sorted worst to best
    class_acc = {}
    for cls in classes:
        t = class_total.get(cls, 0)
        c = class_correct.get(cls, 0)
        class_acc[cls] = (c / t * 100) if t > 0 else 0

    sorted_acc = sorted(class_acc.items(), key=lambda x: x[1])

    print(f"\n{'='*70}")
    print("PER-CLASS ACCURACY (worst to best)")
    print(f"{'='*70}")

    failing = []
    weak = []
    good = []

    for cls, acc in sorted_acc:
        t = class_total.get(cls, 0)
        c = class_correct.get(cls, 0)
        if acc < 50:
            marker = "FAIL"
            failing.append(cls)
        elif acc < 75:
            marker = "WEAK"
            weak.append(cls)
        else:
            marker = "  OK"
            good.append(cls)
        print(f"  [{marker}] {cls:15s}: {c:3d}/{t:3d} = {acc:5.1f}%")

    # Top confusion pairs
    print(f"\n{'='*70}")
    print("TOP CONFUSION PAIRS (true → predicted: count)")
    print(f"{'='*70}")

    confusion_pairs = []
    for true_cls, preds in confusion.items():
        for pred_cls, count in preds.items():
            confusion_pairs.append((true_cls, pred_cls, count))

    confusion_pairs.sort(key=lambda x: -x[2])
    for true_cls, pred_cls, count in confusion_pairs[:40]:
        true_acc = class_acc.get(true_cls, 0)
        print(f"  {true_cls:15s} → {pred_cls:15s}: {count:3d}  (true class acc: {true_acc:.0f}%)")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  FAILING (<50%): {len(failing)} classes")
    for cls in failing:
        top_confused = sorted(confusion.get(cls, {}).items(), key=lambda x: -x[1])[:3]
        confused_str = ", ".join(f"{c}({n})" for c, n in top_confused)
        print(f"    {cls}: {class_acc[cls]:.0f}% — confused with: {confused_str}")

    print(f"  WEAK (50-75%):  {len(weak)} classes")
    for cls in weak:
        top_confused = sorted(confusion.get(cls, {}).items(), key=lambda x: -x[1])[:3]
        confused_str = ", ".join(f"{c}({n})" for c, n in top_confused)
        print(f"    {cls}: {class_acc[cls]:.0f}% — confused with: {confused_str}")

    print(f"  GOOD (>75%):    {len(good)} classes")
    print(f"\n  Classes needing data curation: {failing + weak}")

    # Write results to file for curation script
    results = {
        "overall_top1": correct_top1 / total,
        "overall_top5": correct_top5 / total,
        "per_class_accuracy": class_acc,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "failing_classes": failing,
        "weak_classes": weak,
    }
    results_path = MODEL_DIR / "verification_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
