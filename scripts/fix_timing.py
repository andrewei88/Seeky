#!/usr/bin/env python3
"""Fix timing.json files by rescaling phoneme timestamps to match actual speech in audio.

The timing.json files were proportionally scaled from old audio data and don't match
the current ElevenLabs audio files. This script:
1. Detects where speech actually starts/ends in each .m4a using ffmpeg silence detection
2. Linearly rescales all phoneme timestamps to fit the detected speech window
3. Writes updated timing.json files
"""

import json
import subprocess
import re
from pathlib import Path

VOCAB_DIR = Path(__file__).parent.parent / "ARYA" / "Resources" / "Vocabulary"


def get_duration(audio_path: Path) -> float:
    """Get audio duration using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def detect_speech_bounds(audio_path: Path) -> tuple[float, float]:
    """Detect speech start and end times using ffmpeg silence detection.

    Returns (speech_start, speech_end).

    Logic:
    - Leading silence: a silence event starting near t=0 → speech_start = silence_end
    - Trailing silence: a silence event whose silence_end is near total_duration → speech_end = silence_start
    - If no leading silence detected, speech_start = 0
    - If no trailing silence detected, speech_end = total_duration
    """
    total_duration = get_duration(audio_path)

    result = subprocess.run(
        ["ffmpeg", "-i", str(audio_path), "-af", "silencedetect=noise=-30dB:d=0.05", "-f", "null", "-"],
        capture_output=True, text=True
    )
    stderr = result.stderr

    # Parse silence events as ordered pairs: (silence_start, silence_end)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", stderr)]

    # Build silence intervals
    intervals = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else total_duration
        intervals.append((s, e))

    speech_start = 0.0
    speech_end = total_duration

    # Check for leading silence: starts near 0
    if intervals and intervals[0][0] < 0.05:
        speech_start = intervals[0][1]

    # Check for trailing silence: ends near total_duration
    if intervals and intervals[-1][1] >= total_duration - 0.05:
        speech_end = intervals[-1][0]

    # Sanity checks
    if speech_end <= speech_start:
        # Fallback: assume speech spans 0 to total_duration
        speech_start = 0.0
        speech_end = total_duration

    if speech_end - speech_start < 0.1:
        # Speech window too small, something went wrong
        speech_start = 0.0
        speech_end = total_duration

    return speech_start, speech_end


def rescale_timing(timing_data: dict, speech_start: float, speech_end: float) -> dict:
    """Rescale phoneme timestamps to fit within the detected speech window."""
    phonemes = timing_data["phonemes"]
    if not phonemes:
        return timing_data

    old_start = phonemes[0]["start"]
    old_end = phonemes[-1]["end"]
    old_span = old_end - old_start

    if old_span <= 0:
        return timing_data

    new_span = speech_end - speech_start

    new_phonemes = []
    for p in phonemes:
        new_p_start = speech_start + (p["start"] - old_start) / old_span * new_span
        new_p_end = speech_start + (p["end"] - old_start) / old_span * new_span
        new_phonemes.append({
            "phoneme": p["phoneme"],
            "letters": p["letters"],
            "start": round(new_p_start, 3),
            "end": round(new_p_end, 3),
        })

    return {
        "word": timing_data["word"],
        "phonemes": new_phonemes,
    }


def main():
    # First, restore original timing data from git so we start from the known-good proportions
    import os
    os.system("git checkout HEAD -- ARYA/Resources/Vocabulary/*/timing.json")
    print("Restored original timing.json files from git\n")

    word_dirs = sorted([d for d in VOCAB_DIR.iterdir() if d.is_dir()])

    fixed = 0
    errors = 0
    flagged = []

    for word_dir in word_dirs:
        word = word_dir.name
        audio_path = word_dir / "audio.m4a"
        timing_path = word_dir / "timing.json"

        if not audio_path.exists() or not timing_path.exists():
            print(f"  SKIP {word}: missing audio or timing")
            continue

        try:
            total_dur = get_duration(audio_path)
            speech_start, speech_end = detect_speech_bounds(audio_path)
        except Exception as e:
            print(f"  ERROR {word}: {e}")
            errors += 1
            continue

        with open(timing_path) as f:
            timing_data = json.load(f)

        old_start = timing_data["phonemes"][0]["start"]
        old_end = timing_data["phonemes"][-1]["end"]

        new_timing = rescale_timing(timing_data, speech_start, speech_end)

        new_start = new_timing["phonemes"][0]["start"]
        new_end = new_timing["phonemes"][-1]["end"]

        flag = ""
        if speech_end - speech_start < 0.15:
            flag = " ⚠️ SHORT"
            flagged.append(word)
        elif speech_start == 0 and speech_end == total_dur:
            flag = " (no silence detected)"

        print(f"  {word:15s}: speech=[{speech_start:.3f}-{speech_end:.3f}] "
              f"old=[{old_start:.3f}-{old_end:.3f}] → new=[{new_start:.3f}-{new_end:.3f}] "
              f"(audio={total_dur:.3f}s){flag}")

        with open(timing_path, "w") as f:
            json.dump(new_timing, f, indent=2)
            f.write("\n")

        fixed += 1

    print(f"\nFixed {fixed} words, {errors} errors")
    if flagged:
        print(f"Flagged (may need manual review): {flagged}")


if __name__ == "__main__":
    main()
