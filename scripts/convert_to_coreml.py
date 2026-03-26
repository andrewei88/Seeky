#!/usr/bin/env python3
"""Convert trained ARYA FastViT-T12 to CoreML format.

Produces two CoreML models:
  1. ARYAClassifier.mlpackage — full model (classify + feature vector)
  2. ARYAClassifier_int8.mlpackage — INT8 quantized version

Includes verification: compares PyTorch vs CoreML outputs on a test image.

Usage:
    pip install coremltools torch timm Pillow numpy
    python scripts/convert_to_coreml.py

Input:  models/arya_classifier_best.pth + models/arya_classes.json
Output: ARYA/Resources/ARYAClassifier.mlpackage
"""

import json
from pathlib import Path

import coremltools as ct
import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESOURCES_DIR = PROJECT_ROOT / "ARYA" / "Resources"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Must match train_classifier.py
BACKBONE = "fastvit_t12"


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


class ARYAClassifierForExport(nn.Module):
    """Wrapper that outputs (probabilities, normalized_features) for CoreML."""

    def __init__(self, model: ARYAClassifier):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, features = self.model(x)
        probabilities = torch.softmax(logits, dim=1)
        normalized_features = nn.functional.normalize(features, p=2, dim=1)
        return probabilities, normalized_features


def load_model(num_classes: int) -> ARYAClassifier:
    """Load trained PyTorch model."""
    model = ARYAClassifier(num_classes)
    weights_path = MODEL_DIR / "arya_classifier_best.pth"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def convert_to_coreml(model: ARYAClassifierForExport, classes: list[str]):
    """Convert with normalization baked into the model."""

    class NormalizedModel(nn.Module):
        """Wraps model with ImageNet normalization from 0-1 input."""
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        def forward(self, x):
            x = (x - self.mean) / self.std
            return self.inner(x)

    normalized = NormalizedModel(model)
    normalized.eval()

    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced = torch.jit.trace(normalized, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name="probabilities"),
            ct.TensorType(name="features"),
        ],
        minimum_deployment_target=ct.target.iOS17,
    )

    # Add metadata
    mlmodel.author = "ARYA"
    mlmodel.short_description = (
        f"FastViT-T12 fine-tuned for {len(classes)}-word children's vocabulary. "
        "Outputs class probabilities and 1024-dim feature vector."
    )
    mlmodel.input_description["image"] = "224x224 RGB image of an object"
    mlmodel.output_description["probabilities"] = f"Probability for each of {len(classes)} classes"
    mlmodel.output_description["features"] = "1024-dim L2-normalized feature vector for CorrectionStore"

    mlmodel.user_defined_metadata["classes"] = json.dumps(classes)

    return mlmodel


def verify_outputs(pytorch_model: ARYAClassifierForExport, coreml_model, classes: list[str]):
    """Compare PyTorch and CoreML outputs on a synthetic test image."""
    print("\nVerifying PyTorch vs CoreML output consistency...")

    np.random.seed(42)
    test_pixels = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_pixels, "RGB")

    # PyTorch inference
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    pt_input = transform(test_image).unsqueeze(0)
    with torch.no_grad():
        pt_probs, pt_features = pytorch_model(pt_input)
    pt_probs = pt_probs.numpy().flatten()
    pt_features = pt_features.numpy().flatten()

    # CoreML inference
    cm_output = coreml_model.predict({"image": test_image})
    cm_probs = cm_output["probabilities"].flatten()
    cm_features = cm_output["features"].flatten()

    # Compare
    prob_diff = np.max(np.abs(pt_probs - cm_probs))
    feat_diff = np.max(np.abs(pt_features - cm_features))
    cosine_sim = np.dot(pt_features, cm_features) / (np.linalg.norm(pt_features) * np.linalg.norm(cm_features))

    print(f"  Probability max diff: {prob_diff:.6f} (should be < 0.01)")
    print(f"  Feature max diff:     {feat_diff:.6f} (should be < 0.01)")
    print(f"  Feature cosine sim:   {cosine_sim:.6f} (should be > 0.999)")

    pt_top = np.argsort(pt_probs)[-5:][::-1]
    cm_top = np.argsort(cm_probs)[-5:][::-1]
    print(f"  PyTorch top-5: {[classes[i] for i in pt_top]}")
    print(f"  CoreML  top-5: {[classes[i] for i in cm_top]}")

    if prob_diff > 0.01 or feat_diff > 0.01:
        print("\n  WARNING: Large discrepancy between PyTorch and CoreML outputs!")
        print("  Check that ImageNet normalization is applied consistently.")
        return False

    print("  Outputs match within tolerance")
    return True


def quantize_int8(model_path: Path, output_path: Path):
    """Quantize CoreML model to INT8 for smaller size."""
    print(f"\nQuantizing to INT8...")
    model = ct.models.MLModel(str(model_path))

    quantized = ct.compression_utils.affine_quantize_weights(model, mode="linear")
    quantized.save(str(output_path))

    original_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    quant_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"  Original: {original_size / 1024 / 1024:.1f} MB")
    print(f"  INT8:     {quant_size / 1024 / 1024:.1f} MB")
    print(f"  Ratio:    {quant_size / original_size:.1%}")


def main():
    classes_path = MODEL_DIR / "arya_classes.json"
    with open(classes_path) as f:
        classes = json.load(f)
    print(f"Classes: {len(classes)}")

    print("Loading PyTorch model...")
    base_model = load_model(len(classes))
    export_model = ARYAClassifierForExport(base_model)

    print("Converting to CoreML...")
    mlmodel = convert_to_coreml(export_model, classes)

    output_path = RESOURCES_DIR / "ARYAClassifier.mlpackage"
    mlmodel.save(str(output_path))
    print(f"Saved to: {output_path}")

    model_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"Model size: {model_size / 1024 / 1024:.1f} MB")

    verify_outputs(export_model, mlmodel, classes)

    int8_path = MODEL_DIR / "ARYAClassifier_int8.mlpackage"
    quantize_int8(output_path, int8_path)

    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"  Full model:  {output_path}")
    print(f"  INT8 model:  {int8_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
