#!/usr/bin/env python3
"""Pre-compute MobileCLIP S0 text embeddings for vocabulary words.

Requires: pip install huggingface_hub coremltools open_clip_torch

Downloads the S0 text encoder CoreML model from Hugging Face,
tokenizes each vocabulary word as "a photo of a {word}",
runs through the text encoder, L2-normalizes, and saves as binary.
"""

import json
import struct
from pathlib import Path

import numpy as np

VOCAB_PATH = Path(__file__).parent.parent / "Seeky" / "Resources" / "vocabulary.json"
OUTPUT_PATH = Path(__file__).parent.parent / "Seeky" / "Resources" / "text_embeddings.bin"

def main():
    with open(VOCAB_PATH) as f:
        vocabulary = json.load(f)

    words = [entry["word"] for entry in vocabulary]
    prompts = [f"a photo of a {word}" for word in words]
    print(f"Generating embeddings for {len(words)} words...")

    # Download text encoder if not already cached
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="apple/coreml-mobileclip",
        allow_patterns=["mobileclip_s0_text.mlpackage/**"],
        local_dir="/tmp/mobileclip_download"
    )

    import coremltools as ct
    text_model = ct.models.MLModel("/tmp/mobileclip_download/mobileclip_s0_text.mlpackage")

    # Tokenize using standard CLIP BPE tokenizer (via open_clip)
    import open_clip
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    tokens = tokenizer(prompts)  # shape: [N, 77]
    print(f"Tokens shape: {tokens.shape}")

    # Run each prompt through CoreML text encoder
    all_embeddings = []
    for i, prompt in enumerate(prompts):
        token_array = tokens[i:i+1].numpy().astype(np.int32)
        prediction = text_model.predict({"text": token_array})
        embedding = prediction["final_emb_1"].flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        all_embeddings.append(embedding)

        if i % 20 == 0:
            print(f"  {i}/{len(words)}: '{words[i]}' → dim={len(embedding)}")

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save as binary (float32 array)
    with open(OUTPUT_PATH, "wb") as f:
        for emb in embeddings:
            f.write(struct.pack(f"{len(emb)}f", *emb))

    import os
    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"Saved {len(words)} embeddings ({embeddings.shape[1]}D) to {OUTPUT_PATH}")
    print(f"File size: {file_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
