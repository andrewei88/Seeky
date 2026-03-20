#!/usr/bin/env python3
"""Generate audio files for each vocabulary word using macOS 'say' command.

This is a free alternative to ElevenLabs that works offline.
Uses Samantha voice (high-quality US English) with slow rate for children.
Output: m4a files suitable for the app bundle.

To use ElevenLabs instead (higher quality, requires paid plan):
  export ELEVENLABS_API_KEY="your-key"
  python3 generate_audio.py
"""

import json
import os
import subprocess
from pathlib import Path

VOCAB_PATH = Path(__file__).parent.parent / "ARYA" / "Resources" / "vocabulary.json"
OUTPUT_DIR = Path(__file__).parent.parent / "ARYA" / "Resources" / "Vocabulary"

# macOS voice settings
VOICE = "Samantha"
RATE = 120  # Words per minute (default ~200, slower for children)


def main():
    with open(VOCAB_PATH) as f:
        vocabulary = json.load(f)

    generated = 0
    skipped = 0

    for entry in vocabulary:
        word = entry["word"]
        word_dir = OUTPUT_DIR / word
        word_dir.mkdir(parents=True, exist_ok=True)

        audio_path = word_dir / "audio.m4a"
        if audio_path.exists():
            print(f"  Skipping {word} (already exists)")
            skipped += 1
            continue

        print(f"  Generating: {word}")

        # Generate AIFF with macOS say command
        aiff_path = word_dir / "audio.aiff"
        result = subprocess.run(
            ["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff_path), word],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  ERROR: say failed for {word}: {result.stderr}")
            continue

        # Convert AIFF to M4A using ffmpeg
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(aiff_path),
                "-c:a", "aac", "-b:a", "128k",
                str(audio_path),
                "-y", "-loglevel", "quiet",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  ERROR: ffmpeg failed for {word}: {result.stderr}")
            continue

        # Clean up intermediate file
        aiff_path.unlink(missing_ok=True)

        generated += 1
        print(f"  Done: {word}")

    print(f"\nGenerated: {generated}, Skipped: {skipped}, Total: {len(vocabulary)}")


if __name__ == "__main__":
    main()
