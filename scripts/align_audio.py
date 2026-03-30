#!/usr/bin/env python3
"""Run Montreal Forced Aligner on generated audio files."""

import json
import subprocess
import tempfile
from pathlib import Path

VOCAB_PATH = Path(__file__).parent.parent / "Seeky" / "Resources" / "vocabulary.json"
AUDIO_DIR = Path(__file__).parent.parent / "Seeky" / "Resources" / "Vocabulary"

def main():
    with open(VOCAB_PATH) as f:
        vocabulary = json.load(f)

    for entry in vocabulary:
        word = entry["word"]
        audio_path = AUDIO_DIR / word / "audio.m4a"

        if not audio_path.exists():
            print(f"  Skipping {word} (no audio file)")
            continue

        # MFA needs a .txt file with the transcript alongside the audio
        # Create a temporary directory with the audio and transcript
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Convert m4a to wav for MFA
            wav_path = tmp / f"{word}.wav"
            subprocess.run(
                ["ffmpeg", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(wav_path), "-y", "-loglevel", "quiet"],
                check=True,
            )

            # Create transcript file
            txt_path = tmp / f"{word}.txt"
            txt_path.write_text(word)

            # Output directory
            output_dir = tmp / "output"
            output_dir.mkdir()

            # Run MFA
            result = subprocess.run(
                [
                    "mfa", "align",
                    str(tmp),
                    "english_us_arpa",
                    "english_us_arpa",
                    str(output_dir),
                    "--clean",
                    "--single_speaker",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"  MFA failed for {word}: {result.stderr}")
                continue

            # Copy TextGrid to audio directory
            textgrid_path = output_dir / f"{word}.TextGrid"
            if textgrid_path.exists():
                dest = AUDIO_DIR / word / "aligned.TextGrid"
                dest.write_text(textgrid_path.read_text())
                print(f"  Aligned: {word}")
            else:
                print(f"  No TextGrid output for {word}")

    print("\nAlignment complete")

if __name__ == "__main__":
    main()
