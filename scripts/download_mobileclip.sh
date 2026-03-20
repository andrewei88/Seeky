#!/usr/bin/env bash
# Downloads MobileCLIP S2 CoreML model from Hugging Face

set -e

MODEL_DIR="$(dirname "$0")/../ARYA/Resources"
mkdir -p "$MODEL_DIR"

echo "Downloading MobileCLIP S2 image encoder (CoreML)..."

# Clone just the CoreML model files from Hugging Face
pip install huggingface_hub 2>/dev/null

python3 -c "
from huggingface_hub import hf_hub_download
import shutil

# Download the image encoder CoreML model
path = hf_hub_download(
    repo_id='apple/coreml-mobileclip',
    filename='MobileCLIP-S2-ImageEncoder.mlpackage.zip',
    local_dir='$MODEL_DIR/tmp'
)
print(f'Downloaded to: {path}')
"

# Unzip and place model
cd "$MODEL_DIR/tmp"
unzip -o MobileCLIP-S2-ImageEncoder.mlpackage.zip -d "$MODEL_DIR/"
rm -rf "$MODEL_DIR/tmp"

echo "MobileCLIP model ready at $MODEL_DIR/MobileCLIPImageEncoder.mlpackage"
