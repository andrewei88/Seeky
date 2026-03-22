#!/usr/bin/env bash
# Downloads MobileCLIP S0 CoreML image encoder from Hugging Face

set -e

MODEL_DIR="$(dirname "$0")/../ARYA/Resources"
mkdir -p "$MODEL_DIR"

echo "Downloading MobileCLIP S0 image encoder (CoreML)..."

pip install huggingface_hub 2>/dev/null

python3 -c "
from huggingface_hub import snapshot_download

# Download only the S0 image encoder (22MB)
path = snapshot_download(
    repo_id='apple/coreml-mobileclip',
    allow_patterns=['mobileclip_s0_image.mlpackage/**'],
    local_dir='/tmp/mobileclip_download'
)
print(f'Downloaded to: {path}')

import shutil
shutil.copytree(
    '/tmp/mobileclip_download/mobileclip_s0_image.mlpackage',
    '$MODEL_DIR/MobileCLIPImageEncoder.mlpackage',
    dirs_exist_ok=True
)
"

echo "MobileCLIP S0 image encoder ready at $MODEL_DIR/MobileCLIPImageEncoder.mlpackage"
