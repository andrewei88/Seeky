#!/usr/bin/env python3
"""Train FastViT-T12 for Seeky's vocabulary.

Two-phase training:
  Phase 1: Freeze backbone, train classifier head (5 epochs)
  Phase 2: Unfreeze all, fine-tune end-to-end (15 epochs)

Dual output:
  1. N-class softmax probabilities (for classification)
  2. 1024-dim feature vector from penultimate layer (for CorrectionStore)

Usage:
    pip install torch torchvision timm
    python scripts/train_classifier.py

Input:  data/seeky_training/{train,val}/{word}/*.jpg
Output: models/seeky_classifier.pth (PyTorch weights)
        models/seeky_classes.json (ordered class list)
"""

import json
import os
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "seeky_training"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparams
BATCH_SIZE = 64
NUM_WORKERS = 4
PHASE1_EPOCHS = 5       # frozen backbone
PHASE2_EPOCHS = 25      # full fine-tune (longer for heavier augmentations)
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224

# Backbone: FastViT-T12 (Apple, 6.7M params, 79.3% ImageNet, 1024-dim features)
BACKBONE = "fastvit_t12"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SeekyClassifier(nn.Module):
    """FastViT-T12 with dual output: class probabilities + feature vector.

    The feature vector is the 1024-dim output of the backbone's global average pooling,
    used by CorrectionStore for embedding-based correction.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        # Load pretrained backbone as feature extractor (num_classes=0 removes head)
        self.backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
        self.feat_dim = self.backbone.num_features  # 1024 for fastvit_t12

        # Classification head: feat_dim -> num_classes
        self.class_head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)           # [B, 1024]
        logits = self.class_head(features)    # [B, num_classes]
        return logits, features

    def forward_classify(self, x):
        """Classification only — returns probabilities."""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=1)

    def forward_features(self, x):
        """Feature extraction only — returns L2-normalized 1024-dim vector."""
        _, features = self.forward(x)
        return nn.functional.normalize(features, p=2, dim=1)


def compute_class_weights(dataset, num_classes, device):
    """Compute inverse-frequency class weights, clamped to [0.5, 3.0].

    Boosts rare/weak classes without destabilizing training on common classes.
    """
    counts = Counter()
    for _, label in dataset.samples:
        counts[label] += 1

    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        freq = counts.get(i, 1) / total
        # inverse frequency, normalized so mean weight = 1.0
        w = (1.0 / num_classes) / freq
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    # Clamp to avoid extreme weights on tiny classes
    weights = weights.clamp(min=0.5, max=3.0)
    # Normalize so mean = 1.0
    weights = weights / weights.mean()
    return weights.to(device)


def get_data_loaders():
    """Create train and val data loaders with augmentation.

    Augmentations designed to bridge web-image → phone-camera distribution gap:
    - RandomResizedCrop with wider scale range (objects at varying distances)
    - RandomPerspective (phone held at different angles)
    - GaussianBlur (camera focus issues, motion blur)
    - RandomErasing (partial occlusion from clutter, hands, other objects)
    - Strong ColorJitter (household lighting varies from dim to bright, warm to cool)
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.RandomRotation(20),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
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
    classes = [c.replace("_", " ") for c in train_dataset.classes]
    with open(MODEL_DIR / "seeky_classes.json", "w") as f:
        json.dump(classes, f, indent=2)
    # Also copy to app Resources so the bundle stays in sync
    resources_dir = PROJECT_ROOT / "Seeky" / "Resources"
    import shutil
    shutil.copy2(MODEL_DIR / "seeky_classes.json", resources_dir / "seeky_classes.json")
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
    return train_loader, val_loader, len(classes), train_dataset


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

    train_loader, val_loader, num_classes, train_dataset = get_data_loaders()

    model = SeekyClassifier(num_classes).to(device)
    print(f"Backbone: {BACKBONE} ({model.feat_dim}-dim features)")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Class-weighted loss with label smoothing to boost underrepresented classes
    # and reduce overconfidence on web images (improves generalization to phone camera)
    class_weights = compute_class_weights(train_dataset, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    print(f"Class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}, mean={class_weights.mean():.2f}")
    print(f"Label smoothing: 0.1")

    # Unweighted criterion for validation (fair comparison)
    val_criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Freeze backbone, train head ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: Train classifier head ({PHASE1_EPOCHS} epochs, backbone frozen)")
    print(f"{'='*60}")

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.class_head.parameters(), lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    for epoch in range(PHASE1_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_top5 = validate(model, val_loader, val_criterion, device)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1}/{PHASE1_EPOCHS} ({elapsed:.0f}s) — "
              f"Train: {train_loss:.4f}/{train_acc:.1%} | "
              f"Val: {val_loss:.4f}/{val_acc:.1%} (top5: {val_top5:.1%})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "seeky_classifier_best.pth")

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
        val_loss, val_acc, val_top5 = validate(model, val_loader, val_criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{PHASE2_EPOCHS} ({elapsed:.0f}s, lr={lr:.2e}) — "
              f"Train: {train_loss:.4f}/{train_acc:.1%} | "
              f"Val: {val_loss:.4f}/{val_acc:.1%} (top5: {val_top5:.1%})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "seeky_classifier_best.pth")
            print(f"    New best: {val_acc:.1%}")

    # Save final model too
    torch.save(model.state_dict(), MODEL_DIR / "seeky_classifier_final.pth")

    # Auto-save versioned checkpoint
    versions_dir = MODEL_DIR / "versions"
    versions_dir.mkdir(exist_ok=True)
    existing = sorted(versions_dir.glob("seeky_classifier_v*.pth"))
    if existing:
        last_num = int(existing[-1].stem.split("_v")[1].split("_")[0])
        next_num = last_num + 1
    else:
        next_num = 0
    acc_str = f"{best_val_acc * 100:.1f}pct"
    version_path = versions_dir / f"seeky_classifier_v{next_num}_{acc_str}.pth"
    import shutil as _shutil
    _shutil.copy2(MODEL_DIR / "seeky_classifier_best.pth", version_path)
    print(f"\n  Versioned checkpoint: {version_path}")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Backbone: {BACKBONE}")
    print(f"  Best validation accuracy: {best_val_acc:.1%}")
    print(f"  Model saved to: {MODEL_DIR / 'seeky_classifier_best.pth'}")
    print(f"  Versioned copy: {version_path}")
    print(f"  Classes saved to: {MODEL_DIR / 'seeky_classes.json'}")
    print(f"{'='*60}")
    print(f"\nNext: python scripts/convert_to_coreml.py")


if __name__ == "__main__":
    main()
