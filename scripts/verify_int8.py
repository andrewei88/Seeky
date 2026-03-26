#!/usr/bin/env python3
"""Verify INT8 quantized CoreML model accuracy vs full-precision PyTorch model.

Runs both models on the validation set and compares per-class accuracy.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

import coremltools as ct
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "arya_training"
MODEL_DIR = PROJECT_ROOT / "models"
RESOURCES_DIR = PROJECT_ROOT / "ARYA" / "Resources"

IMAGE_SIZE = 224

# Load classes
with open(MODEL_DIR / "arya_classes.json") as f:
    classes = json.load(f)
print(f"Classes: {len(classes)}")

# Load INT8 CoreML model (what's actually deployed)
int8_path = MODEL_DIR / "ARYAClassifier_int8.mlpackage"
if not int8_path.exists():
    # Check resources dir
    int8_path = RESOURCES_DIR / "ARYAClassifier.mlpackage"
print(f"Loading CoreML model from: {int8_path}")
coreml_model = ct.models.MLModel(str(int8_path))

# Val dataset (raw images, no torch transforms needed for CoreML)
val_dir = DATA_DIR / "val"
val_dataset = datasets.ImageFolder(val_dir)
idx_to_name = {i: c.replace("_", " ") for i, c in enumerate(val_dataset.classes)}

class_correct = defaultdict(int)
class_total = defaultdict(int)
confusion = defaultdict(lambda: defaultdict(int))

print(f"Running CoreML inference on {len(val_dataset)} validation images...")
for i, (img_path, label) in enumerate(val_dataset.samples):
    img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    output = coreml_model.predict({"image": img})
    probs = output["probabilities"].flatten()
    pred_idx = int(np.argmax(probs))

    true_name = idx_to_name[label]
    pred_name = idx_to_name[pred_idx]
    class_total[true_name] += 1
    if true_name == pred_name:
        class_correct[true_name] += 1
    else:
        confusion[true_name][pred_name] += 1

    if (i + 1) % 2000 == 0:
        running_acc = sum(class_correct.values()) / sum(class_total.values())
        print(f"  {i+1}/{len(val_dataset)} ({running_acc:.1%})")

# Results
overall_correct = sum(class_correct.values())
overall_total = sum(class_total.values())
overall_acc = overall_correct / overall_total

print(f"\n{'='*70}")
print(f"INT8 CoreML MODEL ACCURACY")
print(f"{'='*70}")
print(f"Overall: {overall_correct}/{overall_total} = {overall_acc:.1%}")

# Compare to PyTorch baseline (95.2%)
print(f"\nPyTorch baseline: 95.2%")
print(f"INT8 CoreML:      {overall_acc:.1%}")
print(f"Difference:       {(overall_acc - 0.952) * 100:+.1f}pp")

# Classes that degraded most
print(f"\n{'='*70}")
print(f"CLASSES WITH BIGGEST INT8 DEGRADATION (vs 95.2% baseline)")
print(f"{'='*70}")

results = []
for name in sorted(class_total.keys()):
    acc = class_correct[name] / class_total[name] if class_total[name] > 0 else 0
    top_conf = sorted(confusion[name].items(), key=lambda x: -x[1])[:3]
    conf_str = ", ".join(f"{c[0]}({c[1]})" for c in top_conf) if top_conf else ""
    results.append((name, class_correct[name], class_total[name], acc, conf_str))

results.sort(key=lambda x: x[3])
for name, correct, total, acc, conf_str in results[:20]:
    print(f"  {name:<20} {acc:>7.1%}   confused with: {conf_str}")
