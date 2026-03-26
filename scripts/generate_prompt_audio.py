#!/usr/bin/env python3
"""Generate hunt prompt and celebration audio clips using ElevenLabs."""

import os
import sys
from pathlib import Path

from elevenlabs import ElevenLabs

OUTPUT_DIR = Path(__file__).parent.parent / "ARYA" / "Resources" / "Vocabulary" / "_prompts"

PHRASES = {
    "find_the": "Can you find the",
    "yes": "Yes!",
    "thats_right": "That's right!",
    "great_job": "Great job!",
    "you_found_it": "You found it!",
}


def main():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: Set ELEVENLABS_API_KEY environment variable")
        sys.exit(1)

    client = ElevenLabs(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    voice_id = "cgSgspJ2msm6clMCkdW9"  # Jessica

    for filename, text in PHRASES.items():
        audio_path = OUTPUT_DIR / f"{filename}.m4a"
        if audio_path.exists():
            print(f"  Skipping {filename} (already exists)")
            continue

        print(f"  Generating: '{text}' -> {filename}.m4a")

        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.75,
                "similarity_boost": 0.75,
                "speed": 0.7,
            },
        )

        mp3_path = OUTPUT_DIR / f"{filename}.mp3"
        with open(mp3_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)

        os.system(f'ffmpeg -i "{mp3_path}" -c:a aac -b:a 128k "{audio_path}" -y -loglevel quiet')
        mp3_path.unlink()

        print(f"  Done: {filename}")

    print(f"\nGenerated {len(PHRASES)} prompt clips in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
