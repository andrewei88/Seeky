#!/usr/bin/env python3
"""Generate split category prompt audio for consistent chaining.

Instead of single-file category prompts (find_animal.m4a), generates:
  - find_a.m4a / find_an.m4a  (prefix files matching find_the.m4a voice)
  - cat_animal.m4a, cat_kitchen_item.m4a, etc. (category name audio)

These chain at runtime the same way word prompts chain find_the.m4a + word.m4a,
giving consistent voice + pause behavior across all prompts.

Old single-file category prompts (find_animal.m4a etc.) are kept as fallbacks.
"""

import os
import sys
from pathlib import Path

from elevenlabs import ElevenLabs

OUTPUT_DIR = Path(__file__).parent.parent / "Seeky" / "Resources" / "Vocabulary" / "_prompts"

VOICE_ID = "cgSgspJ2msm6clMCkdW9"  # Jessica

# Prefix phrases (same voice/style as find_the.m4a)
PREFIXES = {
    "find_a": "Can you find a",
    "find_an": "Can you find an",
}

# Category names spoken as standalone words (question inflection)
CATEGORY_NAMES = [
    "animal",
    "fruit",
    "food",
    "clothing",
    "kitchen item",
    "furniture",
    "body part",
    "vehicle",
    "toy",
    "bathroom item",
    "school supply",
    "electronics",
]


def generate(client, text: str, output_path: Path):
    """Generate audio via ElevenLabs API and convert to m4a."""
    print(f"  Generating: '{text}' -> {output_path.name}")

    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings={
            "stability": 0.75,
            "similarity_boost": 0.75,
            "speed": 0.7,
        },
    )

    mp3_path = output_path.with_suffix(".mp3")
    with open(mp3_path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)

    os.system(f'ffmpeg -i "{mp3_path}" -c:a aac -b:a 128k "{output_path}" -y -loglevel quiet')
    mp3_path.unlink()
    print(f"  OK: {output_path.name}")


def main():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: Set ELEVENLABS_API_KEY environment variable")
        sys.exit(1)

    client = ElevenLabs(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Generating prefix audio ===")
    for filename, text in PREFIXES.items():
        path = OUTPUT_DIR / f"{filename}.m4a"
        if path.exists():
            print(f"  Skipping {filename}.m4a (already exists)")
            continue
        generate(client, text, path)

    print("\n=== Generating category name audio ===")
    for category in CATEGORY_NAMES:
        filename = f"cat_{category.replace(' ', '_')}"
        path = OUTPUT_DIR / f"{filename}.m4a"
        if path.exists():
            print(f"  Skipping {filename}.m4a (already exists)")
            continue
        generate(client, f"{category}?", path)

    print("\nDone. New files chain like word prompts: find_a.m4a + cat_animal.m4a")


if __name__ == "__main__":
    main()
