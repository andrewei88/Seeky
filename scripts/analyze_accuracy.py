#!/usr/bin/env python3
"""Per-class accuracy analysis on current model's validation set.

Supports both legacy MobileNetV3-Small and current FastViT-T12 backbones.
Auto-detects which model to use based on saved weights.
"""
import json
import sys
import torch
import torch.nn as nn
import timm
from torchvision import datasets, models, transforms
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "arya_training"
MODEL_DIR = PROJECT_ROOT / "models"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Load classes
with open(MODEL_DIR / "arya_classes.json") as f:
    classes = json.load(f)
num_classes = len(classes)
print(f"Classes: {num_classes}")


class ARYAClassifierFastViT(nn.Module):
    """FastViT-T12 backbone (current)."""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("fastvit_t12", pretrained=False, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.class_head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.class_head(features)
        return logits, features


class ARYAClassifierMobileNet(nn.Module):
    """MobileNetV3-Small backbone (legacy)."""
    def __init__(self, num_classes):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.feature_head = nn.Sequential(
            base.classifier[0], base.classifier[1], base.classifier[2],
        )
        self.class_head = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.feature_head(x)
        logits = self.class_head(features)
        return logits, features


def load_model(num_classes, device):
    """Auto-detect and load the correct model architecture."""
    state = torch.load(MODEL_DIR / "arya_classifier_best.pth", map_location=device, weights_only=True)
    # Detect architecture by checking for backbone vs features keys
    if any(k.startswith("backbone.") for k in state.keys()):
        print("Detected FastViT-T12 backbone")
        model = ARYAClassifierFastViT(num_classes).to(device)
    else:
        print("Detected MobileNetV3-Small backbone")
        model = ARYAClassifierMobileNet(num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    model = load_model(num_classes, device)

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    idx_to_name = {i: c.replace("_", " ") for i, c in enumerate(val_dataset.classes)}

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    print("Running validation...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            logits, _ = model(inputs)
            _, predicted = logits.max(1)

            for i in range(labels.size(0)):
                true_name = idx_to_name[labels[i].item()]
                pred_name = idx_to_name[predicted[i].item()]
                class_total[true_name] += 1
                if true_name == pred_name:
                    class_correct[true_name] += 1
                else:
                    confusion[true_name][pred_name] += 1

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(val_loader)}")

    # Results sorted by accuracy (worst first)
    results = []
    for name in sorted(class_total.keys()):
        acc = class_correct[name] / class_total[name] if class_total[name] > 0 else 0
        top_conf = sorted(confusion[name].items(), key=lambda x: -x[1])[:3]
        conf_str = ", ".join(f"{c[0]}({c[1]})" for c in top_conf) if top_conf else ""
        results.append((name, class_correct[name], class_total[name], acc, conf_str))

    results.sort(key=lambda x: x[3])

    print(f"\n{'='*90}")
    print(f"PER-CLASS ACCURACY (worst first)")
    print(f"{'='*90}")
    print(f"  {'Class':<20} {'Correct':>8} {'Total':>8} {'Acc':>8}   Top confusions")
    print(f"  {'-'*86}")

    for name, correct, total, acc, conf_str in results:
        print(f"  {name:<20} {correct:>6}/{total:<6} {acc:>7.1%}   {conf_str}")

    overall_correct = sum(class_correct.values())
    overall_total = sum(class_total.values())
    print(f"\n  Overall: {overall_correct}/{overall_total} = {overall_correct/overall_total:.1%}")

    # Bottom 30
    print(f"\n{'='*90}")
    print(f"BOTTOM 30 CLASSES")
    print(f"{'='*90}")
    for name, correct, total, acc, conf_str in results[:30]:
        print(f"  {name:<20} {acc:>7.1%}   confused with: {conf_str}")

    # Common household objects
    household = {"cup", "bowl", "plate", "spoon", "fork", "bottle", "glass", "chair", "table",
                 "couch", "bed", "door", "window", "light", "lamp", "tv", "monitor", "laptop",
                 "phone", "keyboard", "shoe", "sock", "ball", "book", "doll", "bag", "box",
                 "paper", "soap", "towel", "blanket", "pillow", "clock", "mirror"}

    print(f"\n{'='*90}")
    print(f"COMMON HOUSEHOLD OBJECTS")
    print(f"{'='*90}")
    household_results = [(n, c, t, a, cf) for n, c, t, a, cf in results if n in household]
    household_results.sort(key=lambda x: x[3])
    for name, correct, total, acc, conf_str in household_results:
        print(f"  {name:<20} {acc:>7.1%}   confused with: {conf_str}")

    # Top confusion pairs (bidirectional)
    print(f"\n{'='*90}")
    print(f"TOP CONFUSION PAIRS")
    print(f"{'='*90}")
    pairs = defaultdict(int)
    for true_name, preds in confusion.items():
        for pred_name, count in preds.items():
            key = tuple(sorted([true_name, pred_name]))
            pairs[key] += count

    for (a, b), count in sorted(pairs.items(), key=lambda x: -x[1])[:20]:
        print(f"  {a} <-> {b}: {count} confusions")


if __name__ == "__main__":
    main()
