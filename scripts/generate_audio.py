#!/usr/bin/env python3
"""Generate audio files for each vocabulary word using ElevenLabs."""

import json
import os
import sys
from pathlib import Path

from elevenlabs import ElevenLabs

VOCAB_PATH = Path(__file__).parent.parent / "ARYA" / "Resources" / "vocabulary.json"
OUTPUT_DIR = Path(__file__).parent.parent / "ARYA" / "Resources" / "Vocabulary"

def main():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: Set ELEVENLABS_API_KEY environment variable")
        sys.exit(1)

    client = ElevenLabs(api_key=api_key)

    with open(VOCAB_PATH) as f:
        vocabulary = json.load(f)

    # Use a warm, friendly voice suitable for toddlers.
    voice_id = "cgSgspJ2msm6clMCkdW9"  # Jessica — Playful, Bright, Warm

    for entry in vocabulary:
        word = entry["word"]
        word_dir = OUTPUT_DIR / word
        word_dir.mkdir(parents=True, exist_ok=True)

        audio_path = word_dir / "audio.m4a"
        if audio_path.exists():
            print(f"  Skipping {word} (already exists)")
            continue

        print(f"  Generating: {word}")

        audio_generator = client.text_to_speech.convert(
            text=word,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.75,
                "similarity_boost": 0.75,
                "speed": 0.7,
            },
        )

        # Save as mp3 first, then convert to m4a
        mp3_path = word_dir / "audio.mp3"
        with open(mp3_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)

        # Convert mp3 to m4a using ffmpeg
        os.system(f'ffmpeg -i "{mp3_path}" -c:a aac -b:a 128k "{audio_path}" -y -loglevel quiet')
        mp3_path.unlink()

        print(f"  Done: {word}")

    print(f"\nGenerated audio for {len(vocabulary)} words")

if __name__ == "__main__":
    main()
