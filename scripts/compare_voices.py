#!/usr/bin/env python3
"""Compare ElevenLabs voices for phoneme clarity.

Generates test samples with different voices so you can listen and pick
the best one for teaching children to read.

Usage:
    python scripts/compare_voices.py --list          # list available voices
    python scripts/compare_voices.py --compare       # generate comparison samples
    python scripts/compare_voices.py --compare --voices "id1,id2,id3"  # specific voices
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from elevenlabs import ElevenLabs

OUTPUT_DIR = Path(__file__).parent.parent / "voice_comparison"

# Test words chosen to cover different phoneme types:
# - stops (t, d, k), fricatives (sh, s), vowels, diphthongs, multi-syllable
TEST_WORDS = ["toaster", "elephant", "butterfly", "shoe", "glass", "microwave"]

# Voices to compare (premade voices available on free tier)
DEFAULT_VOICE_IDS = [
    "cgSgspJ2msm6clMCkdW9",  # Jessica (current) — Playful, Bright, Warm
    "XB0fDUnXU5powFXDhCwa",  # Charlotte — Animated, Seductive
    "EXAVITQu4vr4xnSDxMaL",  # Sarah — Soft, Gentle
    "pFZP5JQG7iQjIQuC4Bku",  # Lily — Warm, Soothing
    "jsCqWAovK2LkecY7zXl4",  # Freya — Mature, Confident
]


def list_voices(client: ElevenLabs):
    """List all available voices with their labels."""
    response = client.voices.get_all()
    voices = response.voices

    print(f"\nAvailable voices ({len(voices)}):\n")
    print(f"{'ID':<30} {'Name':<15} {'Labels'}")
    print("-" * 80)

    for voice in sorted(voices, key=lambda v: v.name or ""):
        labels = voice.labels or {}
        label_str = ", ".join(f"{k}={v}" for k, v in labels.items())
        print(f"{voice.voice_id:<30} {voice.name or '?':<15} {label_str}")


def generate_comparison(client: ElevenLabs, voice_ids: list[str]):
    """Generate test samples with each voice."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get voice names for labels
    all_voices = {v.voice_id: v.name for v in client.voices.get_all().voices}

    for voice_id in voice_ids:
        voice_name = all_voices.get(voice_id, "unknown")
        voice_dir = OUTPUT_DIR / f"{voice_name}_{voice_id[:8]}"
        voice_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Voice: {voice_name} ({voice_id})")
        print(f"{'='*50}")

        for word in TEST_WORDS:
            output_path = voice_dir / f"{word}.m4a"
            if output_path.exists():
                print(f"  {word}: already exists, skipping")
                continue

            print(f"  Generating: {word}...")
            mp3_path = voice_dir / f"{word}.mp3"

            try:
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

                with open(mp3_path, "wb") as f:
                    for chunk in audio_generator:
                        f.write(chunk)

                # Convert to m4a
                subprocess.run(
                    ["ffmpeg", "-i", str(mp3_path), "-y", "-loglevel", "quiet",
                     str(output_path)],
                    check=True,
                )
                mp3_path.unlink()
                print(f"  {word}: done")

            except Exception as e:
                print(f"  {word}: FAILED - {e}")

    print(f"\n\nComparison samples saved to: {OUTPUT_DIR}")
    print("Listen to each voice's samples and pick the one with clearest phoneme articulation.")
    print("\nTo use a different voice, update VOICE_ID in scripts/generate_audio.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List available voices")
    parser.add_argument("--compare", action="store_true", help="Generate comparison samples")
    parser.add_argument("--voices", help="Comma-separated voice IDs to compare")
    args = parser.parse_args()

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: Set ELEVENLABS_API_KEY environment variable")
        sys.exit(1)

    client = ElevenLabs(api_key=api_key)

    if args.list:
        list_voices(client)
    elif args.compare:
        voice_ids = args.voices.split(",") if args.voices else DEFAULT_VOICE_IDS
        generate_comparison(client, voice_ids)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
