#!/usr/bin/env python3
"""Generate timing.json files by analyzing audio duration and distributing
phoneme timings proportionally.

For single-word utterances from macOS 'say', we measure the actual audio
duration, trim leading/trailing silence, and distribute phoneme boundaries
proportionally based on typical phoneme durations.

This produces good-enough timing for letter highlighting. For production
quality, use Montreal Forced Aligner (align_audio.py + build_timing_data.py).
"""

import json
import subprocess
from pathlib import Path

AUDIO_DIR = Path(__file__).parent.parent / "Seeky" / "Resources" / "Vocabulary"
MAPPINGS_PATH = Path(__file__).parent / "phoneme_to_letter_mappings.json"

# Approximate relative durations for phoneme classes (arbitrary units)
# Vowels are longer, stops are shorter
PHONEME_WEIGHTS = {
    # Stops (short)
    "P": 0.6, "B": 0.6, "T": 0.6, "D": 0.6, "K": 0.6, "G": 0.6,
    # Fricatives (medium)
    "F": 0.8, "V": 0.8, "S": 0.8, "Z": 0.8, "SH": 0.9, "ZH": 0.9,
    "TH": 0.8, "DH": 0.8, "HH": 0.7,
    # Affricates
    "CH": 0.8, "JH": 0.8,
    # Nasals
    "M": 0.8, "N": 0.8, "NG": 0.8,
    # Liquids/Glides
    "L": 0.7, "R": 0.7, "W": 0.6, "Y": 0.6,
    # Vowels (longer)
    "AA": 1.2, "AE": 1.2, "AH": 1.0, "AO": 1.2, "AW": 1.3,
    "AY": 1.3, "EH": 1.0, "ER": 1.1, "EY": 1.2, "IH": 0.9,
    "IY": 1.1, "OW": 1.2, "OY": 1.3, "UH": 1.0, "UW": 1.1,
}


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path),
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def main():
    with open(MAPPINGS_PATH) as f:
        letter_mappings = json.load(f)

    generated = 0

    for word, phoneme_map in letter_mappings.items():
        audio_path = AUDIO_DIR / word / "audio.m4a"
        if not audio_path.exists():
            print(f"  Skipping {word} (no audio file)")
            continue

        duration = get_audio_duration(audio_path)

        # Trim estimated silence padding (say adds ~0.05s lead-in)
        speech_start = 0.05
        speech_end = max(duration - 0.05, speech_start + 0.1)
        speech_duration = speech_end - speech_start

        # Calculate weighted phoneme durations
        weights = []
        for entry in phoneme_map:
            w = PHONEME_WEIGHTS.get(entry["phoneme"], 1.0)
            weights.append(w)

        total_weight = sum(weights)

        # Distribute time proportionally
        timing_phonemes = []
        current_time = speech_start

        for i, entry in enumerate(phoneme_map):
            phoneme_duration = (weights[i] / total_weight) * speech_duration
            start = round(current_time, 3)
            end = round(current_time + phoneme_duration, 3)

            timing_phonemes.append({
                "phoneme": entry["phoneme"],
                "letters": entry["letters"],
                "start": start,
                "end": end,
            })

            current_time = end

        timing_data = {
            "word": word,
            "phonemes": timing_phonemes,
        }

        output_path = AUDIO_DIR / word / "timing.json"
        with open(output_path, "w") as f:
            json.dump(timing_data, f, indent=2)

        generated += 1
        print(f"  Built timing: {word} ({len(timing_phonemes)} phonemes, {duration:.2f}s)")

    print(f"\nGenerated timing for {generated} words")


if __name__ == "__main__":
    main()
