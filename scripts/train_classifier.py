#!/usr/bin/env python3
"""Train MobileNetV3-Small for ARYA's 107-word vocabulary.

Two-phase training:
  Phase 1: Freeze backbone, train classifier head (5 epochs)
  Phase 2: Unfreeze all, fine-tune end-to-end (15 epochs)

Dual output:
  1. 107-class softmax probabilities (for classification)
  2. 1024-dim feature vector from penultimate layer (for CorrectionStore)

Usage:
    pip install torch torchvision
    python scripts/train_classifier.py

Input:  data/arya_training/{train,val}/{word}/*.jpg
Output: models/arya_classifier.pth (PyTorch weights)
        models/arya_classes.json (ordered class list)
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "arya_training"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparams
BATCH_SIZE = 64
NUM_WORKERS = 4
PHASE1_EPOCHS = 5       # frozen backbone
PHASE2_EPOCHS = 15      # full fine-tune
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224         # MobileNetV3 standard input

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ARYAClassifier(nn.Module):
    """MobileNetV3-Small with dual output: class probabilities + feature vector.

    The feature vector is the 1024-dim output of the penultimate layer,
    used by CorrectionStore for embedding-based correction (same role as
    CLIP embeddings in the Tier 3 architecture).
    """

    def __init__(self, num_classes: int):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Backbone: everything up to the classifier
        self.features = base.features
        self.avgpool = base.avgpool

        # The MobileNetV3 classifier is: Linear(576, 1024) → Hardswish → Dropout → Linear(1024, 1000)
        # We keep the first part (576 → 1024 + activation) as our feature extractor
        self.feature_head = nn.Sequential(
            base.classifier[0],   # Linear(576, 1024)
            base.classifier[1],   # Hardswish
            base.classifier[2],   # Dropout
        )

        # New classification head: 1024 → num_classes
        self.class_head = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)         # [B, 576]
        features = self.feature_head(x)  # [B, 1024]
        logits = self.class_head(features)  # [B, num_classes]
        return logits, features

    def forward_classify(self, x):
        """Classification only — returns probabilities."""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=1)

    def forward_features(self, x):
        """Feature extraction only — returns L2-normalized 1024-dim vector."""
        _, features = self.forward(x)
        return nn.functional.normalize(features, p=2, dim=1)


def get_data_loaders():
    """Create train and val data loaders with augmentation."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),  # resize larger then center crop
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

    # Save class ordering (folder names sorted alphabetically by ImageFolder)
    # Replace underscores with spaces to match vocabulary.json naming convention
    # (e.g., "teddy_bear" folder → "teddy bear" class, "toilet_paper" → "toilet paper")
    classes = [c.replace("_", " ") for c in train_dataset.classes]
    with open(MODEL_DIR / "arya_classes.json", "w") as f:
        json.dump(classes, f, indent=2)
    # Also copy to app Resources so the bundle stays in sync
    resources_dir = PROJECT_ROOT / "ARYA" / "Resources"
    import shutil
    shutil.copy2(MODEL_DIR / "arya_classes.json", resources_dir / "arya_classes.json")
    print(f"Classes ({len(classes)}): {classes[:5]}...{classes[-5:]}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    return train_loader, val_loader, len(classes)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validate, return (loss, accuracy, top5_accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = logits.topk(5, dim=1)
            correct_top5 += sum(labels[i] in top5_pred[i] for i in range(labels.size(0)))

    return running_loss / total, correct / total, correct_top5 / total


def main():
    # Device selection: MPS (Apple Silicon), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU (this will be slow)")

    train_loader, val_loader, num_classes = get_data_loaders()

    model = ARYAClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Freeze backbone, train head ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: Train classifier head ({PHASE1_EPOCHS} epochs, backbone frozen)")
    print(f"{'='*60}")

    # Freeze everything except class_head
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.feature_head.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.class_head.parameters(), lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    for epoch in range(PHASE1_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_top5 = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1}/{PHASE1_EPOCHS} ({elapsed:.0f}s) — "
              f"Train: {train_loss:.4f}/{train_acc:.1%} | "
              f"Val: {val_loss:.4f}/{val_acc:.1%} (top5: {val_top5:.1%})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "arya_classifier_best.pth")

    # ── Phase 2: Unfreeze all, fine-tune ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: Full fine-tune ({PHASE2_EPOCHS} epochs, all layers)")
    print(f"{'='*60}")

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    for epoch in range(PHASE2_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_top5 = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{PHASE2_EPOCHS} ({elapsed:.0f}s, lr={lr:.2e}) — "
              f"Train: {train_loss:.4f}/{train_acc:.1%} | "
              f"Val: {val_loss:.4f}/{val_acc:.1%} (top5: {val_top5:.1%})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "arya_classifier_best.pth")
            print(f"    ✓ New best: {val_acc:.1%}")

    # Save final model too
    torch.save(model.state_dict(), MODEL_DIR / "arya_classifier_final.pth")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Best validation accuracy: {best_val_acc:.1%}")
    print(f"  Model saved to: {MODEL_DIR / 'arya_classifier_best.pth'}")
    print(f"  Classes saved to: {MODEL_DIR / 'arya_classes.json'}")
    print(f"{'='*60}")
    print(f"\nNext: python scripts/convert_to_coreml.py")


if __name__ == "__main__":
    main()
