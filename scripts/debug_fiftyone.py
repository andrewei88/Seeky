#!/usr/bin/env python3
"""Debug: check what fiftyone returns for a few labels."""
import fiftyone as fo
import fiftyone.zoo as foz

# Test with "ball" — one of the words that got 0 images
for label in ["Ball", "Banana", "Bear"]:
    print(f"\n--- Testing label: '{label}' ---")
    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=[label],
            max_samples=5,
            shuffle=True,
            dataset_name=None,
        )
        print(f"  Dataset: {len(dataset)} samples")
        for sample in dataset.head(2):
            print(f"  Sample filepath: {sample.filepath}")
            import os
            print(f"  File exists: {os.path.exists(sample.filepath)}")
            if sample.ground_truth:
                dets = sample.ground_truth.detections
                print(f"  Detections: {len(dets)}")
                for d in dets[:3]:
                    print(f"    label='{d.label}', bbox={d.bounding_box}")
            else:
                # Check other fields
                print(f"  Fields: {sample.field_names}")
                for field_name in sample.field_names:
                    val = sample[field_name]
                    if val is not None and field_name not in ("id", "filepath", "tags", "metadata"):
                        print(f"    {field_name}: {type(val).__name__} = {str(val)[:200]}")
        dataset.delete()
    except Exception as e:
        print(f"  ERROR: {e}")
