#!/usr/bin/env python3
"""Pre-compute MobileCLIP text embeddings for vocabulary words."""

import json
import struct
from pathlib import Path

import torch
import mobileclip

VOCAB_PATH = Path(__file__).parent.parent / "ARYA" / "Resources" / "vocabulary.json"
OUTPUT_PATH = Path(__file__).parent.parent / "ARYA" / "Resources" / "text_embeddings.bin"

def main():
    with open(VOCAB_PATH) as f:
        vocabulary = json.load(f)

    words = [entry["word"] for entry in vocabulary]
    prompts = [f"a photo of a {word}" for word in words]

    # Load MobileCLIP S2 model
    model, _, preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s2", pretrained="checkpoints/mobileclip_s2.pt"
    )
    tokenizer = mobileclip.get_tokenizer("mobileclip_s2")

    # Encode all prompts
    tokens = tokenizer(prompts)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Save as binary (float32 array)
    embeddings = text_features.cpu().numpy()
    with open(OUTPUT_PATH, "wb") as f:
        for embedding in embeddings:
            f.write(struct.pack(f"{len(embedding)}f", *embedding))

    print(f"Saved {len(words)} embeddings ({embeddings.shape[1]}D) to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
