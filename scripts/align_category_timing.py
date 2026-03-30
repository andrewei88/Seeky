#!/usr/bin/env python3
"""Generate timing.json for quiz category names from their audio clips.

Each category has a pre-recorded audio clip at _prompts/cat_{name}.m4a.
This script detects phoneme boundaries using spectral flux (same approach
as align_timing_from_energy.py) and writes timing.json files that the
app can load for letter-by-letter highlighting.

Output goes to Vocabulary/{display_name}/timing.json so that the existing
TimingData.load(word:) works with no Swift code changes.

Usage:
    python scripts/align_category_timing.py
    python scripts/align_category_timing.py --debug animal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse audio analysis functions from the word timing script
sys.path.insert(0, str(Path(__file__).parent))
from align_timing_from_energy import (
    load_audio_pcm,
    compute_energy,
    compute_spectral_flux,
    find_speech_region,
    find_phoneme_boundaries,
)

VOCAB_DIR = Path(__file__).parent.parent / "Seeky" / "Resources" / "Vocabulary"
PROMPTS_DIR = VOCAB_DIR / "_prompts"

# Phoneme-to-letter mappings for each category display name.
# Letter indices are 0-based positions in the display string (spaces included).
CATEGORY_PHONEMES: dict[str, list[dict]] = {
    "animal": [
        {"phoneme": "AE", "letters": [0]},       # a
        {"phoneme": "N", "letters": [1]},         # n
        {"phoneme": "AH", "letters": [2]},        # i
        {"phoneme": "M", "letters": [3]},         # m
        {"phoneme": "AH", "letters": [4]},        # a
        {"phoneme": "L", "letters": [5]},         # l
    ],
    "fruit": [
        {"phoneme": "F", "letters": [0]},         # f
        {"phoneme": "R", "letters": [1]},         # r
        {"phoneme": "UW", "letters": [2, 3]},     # ui
        {"phoneme": "T", "letters": [4]},         # t
    ],
    "food": [
        {"phoneme": "F", "letters": [0]},         # f
        {"phoneme": "UW", "letters": [1, 2]},     # oo
        {"phoneme": "D", "letters": [3]},         # d
    ],
    "clothing": [
        {"phoneme": "K", "letters": [0]},         # c
        {"phoneme": "L", "letters": [1]},         # l
        {"phoneme": "OW", "letters": [2]},        # o
        {"phoneme": "DH", "letters": [3, 4]},     # th
        {"phoneme": "IH", "letters": [5]},        # i
        {"phoneme": "NG", "letters": [6, 7]},     # ng
    ],
    "kitchen item": [
        # "kitchen item" = k(0) i(1) t(2) c(3) h(4) e(5) n(6) ' '(7) i(8) t(9) e(10) m(11)
        {"phoneme": "K", "letters": [0]},         # k
        {"phoneme": "IH", "letters": [1]},        # i
        {"phoneme": "CH", "letters": [2, 3, 4]},  # tch
        {"phoneme": "AH", "letters": [5]},        # e
        {"phoneme": "N", "letters": [6]},         # n
        {"phoneme": "AY", "letters": [8]},        # i
        {"phoneme": "T", "letters": [9]},         # t
        {"phoneme": "AH", "letters": [10]},       # e
        {"phoneme": "M", "letters": [11]},        # m
    ],
    "furniture": [
        # "furniture" = f(0) u(1) r(2) n(3) i(4) t(5) u(6) r(7) e(8)
        {"phoneme": "F", "letters": [0]},         # f
        {"phoneme": "ER", "letters": [1, 2]},     # ur
        {"phoneme": "N", "letters": [3]},         # n
        {"phoneme": "IH", "letters": [4]},        # i
        {"phoneme": "CH", "letters": [5]},        # t (ture -> cher)
        {"phoneme": "ER", "letters": [6, 7, 8]},  # ure
    ],
    "body part": [
        # "body part" = b(0) o(1) d(2) y(3) ' '(4) p(5) a(6) r(7) t(8)
        {"phoneme": "B", "letters": [0]},         # b
        {"phoneme": "AA", "letters": [1]},        # o
        {"phoneme": "D", "letters": [2]},         # d
        {"phoneme": "IY", "letters": [3]},        # y
        {"phoneme": "P", "letters": [5]},         # p
        {"phoneme": "AA", "letters": [6]},        # a
        {"phoneme": "R", "letters": [7]},         # r
        {"phoneme": "T", "letters": [8]},         # t
    ],
    "vehicle": [
        # "vehicle" = v(0) e(1) h(2) i(3) c(4) l(5) e(6)
        {"phoneme": "V", "letters": [0]},         # v
        {"phoneme": "IY", "letters": [1]},        # e
        {"phoneme": "IH", "letters": [2, 3]},     # hi
        {"phoneme": "K", "letters": [4]},         # c
        {"phoneme": "AH", "letters": [5]},        # l
        {"phoneme": "L", "letters": [6]},         # e
    ],
    "toy": [
        {"phoneme": "T", "letters": [0]},         # t
        {"phoneme": "OY", "letters": [1, 2]},     # oy
    ],
    "bathroom item": [
        # "bathroom item" = b(0) a(1) t(2) h(3) r(4) o(5) o(6) m(7) ' '(8) i(9) t(10) e(11) m(12)
        {"phoneme": "B", "letters": [0]},         # b
        {"phoneme": "AE", "letters": [1]},        # a
        {"phoneme": "TH", "letters": [2, 3]},     # th
        {"phoneme": "R", "letters": [4]},         # r
        {"phoneme": "UW", "letters": [5, 6]},     # oo
        {"phoneme": "M", "letters": [7]},         # m
        {"phoneme": "AY", "letters": [9]},        # i
        {"phoneme": "T", "letters": [10]},        # t
        {"phoneme": "AH", "letters": [11]},       # e
        {"phoneme": "M", "letters": [12]},        # m
    ],
    "school supply": [
        # "school supply" = s(0) c(1) h(2) o(3) o(4) l(5) ' '(6) s(7) u(8) p(9) p(10) l(11) y(12)
        {"phoneme": "S", "letters": [0]},         # s
        {"phoneme": "K", "letters": [1, 2]},      # ch
        {"phoneme": "UW", "letters": [3, 4]},     # oo
        {"phoneme": "L", "letters": [5]},         # l
        {"phoneme": "S", "letters": [7]},         # s
        {"phoneme": "AH", "letters": [8]},        # u
        {"phoneme": "P", "letters": [9, 10]},     # pp
        {"phoneme": "L", "letters": [11]},        # l
        {"phoneme": "AY", "letters": [12]},       # y
    ],
}


def align_category(name: str, phoneme_map: list[dict], debug: bool = False) -> dict | None:
    """Generate timing data for a category name from its audio clip."""
    audio_filename = f"cat_{name.replace(' ', '_')}"
    audio_path = PROMPTS_DIR / f"{audio_filename}.m4a"

    if not audio_path.exists():
        print(f"  Skipping {name} (no audio at {audio_path})")
        return None

    sr = 16000
    hop = 160
    frame_len = 512

    samples = load_audio_pcm(audio_path, sr=sr)
    energy = compute_energy(samples, sr=sr, frame_len=frame_len, hop=hop)
    flux = compute_spectral_flux(samples, sr=sr, frame_len=frame_len, hop=hop)

    speech_start, speech_end = find_speech_region(energy, hop=hop, sr=sr)
    start_frame = int(speech_start * sr / hop)
    end_frame = int(speech_end * sr / hop)

    n_phonemes = len(phoneme_map)
    n_boundaries = n_phonemes - 1

    boundaries_sec = find_phoneme_boundaries(
        flux, energy, n_boundaries,
        start_frame, end_frame, hop=hop, sr=sr
    )

    # Build timing entries with minimum duration enforcement
    MIN_PHONEME_MS = 40
    times = [speech_start] + boundaries_sec + [speech_end]

    for _pass in range(3):
        for i in range(len(times) - 1):
            dur = times[i + 1] - times[i]
            if dur < MIN_PHONEME_MS / 1000:
                deficit = MIN_PHONEME_MS / 1000 - dur
                if i > 0 and (i + 1 >= len(times) - 1 or
                              (times[i] - times[i - 1]) > (times[i + 2] - times[i + 1])):
                    times[i] -= deficit / 2
                elif i + 1 < len(times) - 1:
                    times[i + 1] += deficit / 2

    phonemes = []
    for i, entry in enumerate(phoneme_map):
        phonemes.append({
            "phoneme": entry["phoneme"],
            "letters": entry["letters"],
            "start": round(times[i], 3),
            "end": round(times[i + 1], 3),
        })

    if debug:
        print(f"\n  '{name}': speech region {speech_start:.3f}s - {speech_end:.3f}s")
        print(f"  Audio duration: {len(samples)/sr:.3f}s")
        print(f"  Boundaries: {[f'{b:.3f}' for b in boundaries_sec]}")
        for p in phonemes:
            dur = p['end'] - p['start']
            print(f"    {p['phoneme']:4s} [{p['start']:.3f} - {p['end']:.3f}] ({dur*1000:.0f}ms) -> letters {p['letters']}")

    return {
        "word": name,
        "phonemes": phonemes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", nargs="?", const="all",
                        help="Show detailed output (optionally for a specific category)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write files, just show what would change")
    args = parser.parse_args()

    generated = 0

    for name, phoneme_map in CATEGORY_PHONEMES.items():
        do_debug = args.debug in ("all", name)
        result = align_category(name, phoneme_map, debug=do_debug)
        if result is None:
            continue

        # Store timing at Vocabulary/{name}/timing.json so TimingData.load works as-is
        output_dir = VOCAB_DIR / name
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "timing.json"

        if output_path.exists():
            with open(output_path) as f:
                old = json.load(f)
            if old == result:
                if do_debug:
                    print(f"  {name}: unchanged")
                continue

        if args.dry_run:
            print(f"  Would write: {output_path}")
        else:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  {name}: {len(phoneme_map)} phonemes aligned -> {output_path.relative_to(VOCAB_DIR)}")

        generated += 1

    print(f"\nGenerated timing for {generated} categories")


if __name__ == "__main__":
    main()
