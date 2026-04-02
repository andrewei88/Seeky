#!/usr/bin/env python3
"""Generate accurate timing.json by detecting phoneme boundaries from audio energy.

Uses spectral flux onset detection to find transitions between phonemes.
For single-word utterances at slow speed (0.7x), phoneme boundaries are
well-separated and detectable from spectral changes.

Much more accurate than proportional weights, doesn't require MFA.

Usage:
    python scripts/align_timing_from_energy.py
    python scripts/align_timing_from_energy.py --word toaster  # single word
    python scripts/align_timing_from_energy.py --debug toaster  # show plots
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import medfilt

AUDIO_DIR = Path(__file__).parent.parent / "Seeky" / "Resources" / "Vocabulary"
MAPPINGS_PATH = Path(__file__).parent / "phoneme_to_letter_mappings.json"

# Phoneme class properties for boundary refinement
VOICED_PHONEMES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "D", "DH", "EH", "ER", "EY",
    "G", "IH", "IY", "JH", "L", "M", "N", "NG", "OW", "OY", "R", "UH",
    "UW", "V", "W", "Y", "Z", "ZH",
}
SILENCE_PHONEMES = {"P", "T", "K"}  # Unvoiced stops have a silence gap


def load_audio_pcm(audio_path: Path, sr: int = 16000) -> np.ndarray:
    """Load audio as mono float32 PCM using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(audio_path),
            "-ar", str(sr), "-ac", "1",
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-v", "quiet", "-",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {audio_path}")

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def compute_spectral_flux(samples: np.ndarray, sr: int = 16000,
                          frame_len: int = 512, hop: int = 160) -> np.ndarray:
    """Compute spectral flux (half-wave rectified) as onset strength."""
    n_frames = (len(samples) - frame_len) // hop + 1
    if n_frames < 2:
        return np.array([0.0])

    window = np.hanning(frame_len)
    prev_spec = None
    flux = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        frame = samples[start:start + frame_len] * window
        spec = np.abs(np.fft.rfft(frame))

        if prev_spec is not None:
            # Half-wave rectified difference (only increases)
            diff = spec - prev_spec
            flux[i] = np.sum(np.maximum(diff, 0))
        prev_spec = spec

    return flux


def compute_energy(samples: np.ndarray, sr: int = 16000,
                   frame_len: int = 512, hop: int = 160) -> np.ndarray:
    """Compute frame-level energy (RMS)."""
    n_frames = (len(samples) - frame_len) // hop + 1
    energy = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        frame = samples[start:start + frame_len]
        energy[i] = np.sqrt(np.mean(frame ** 2))

    return energy


def find_speech_region(energy: np.ndarray, hop: int = 160, sr: int = 16000,
                       threshold_ratio: float = 0.05):
    """Find speech onset and offset from energy envelope."""
    if len(energy) == 0:
        return 0.0, 0.0

    threshold = np.max(energy) * threshold_ratio

    # Find first frame above threshold
    above = np.where(energy > threshold)[0]
    if len(above) == 0:
        return 0.0, len(energy) * hop / sr

    start_frame = max(0, above[0] - 2)  # small look-back
    end_frame = min(len(energy) - 1, above[-1] + 2)

    return start_frame * hop / sr, end_frame * hop / sr


def find_phoneme_boundaries(flux: np.ndarray, energy: np.ndarray,
                            n_boundaries: int, speech_start_frame: int,
                            speech_end_frame: int, hop: int = 160,
                            sr: int = 16000) -> list[float]:
    """Find N boundary times between phonemes using spectral flux peaks.

    Returns N boundary times (in seconds) that split the speech region
    into N+1 segments corresponding to phonemes.
    """
    if n_boundaries <= 0:
        return []

    # Work only within the speech region
    region_flux = flux[speech_start_frame:speech_end_frame + 1].copy()

    # Smooth to reduce noise
    if len(region_flux) > 5:
        region_flux = medfilt(region_flux, kernel_size=3)

    # Normalize
    max_flux = np.max(region_flux)
    if max_flux > 0:
        region_flux /= max_flux

    # Find peaks (local maxima above a threshold)
    min_distance_frames = int(0.03 * sr / hop)  # 30ms minimum between peaks
    peaks = []
    for i in range(1, len(region_flux) - 1):
        if (region_flux[i] > region_flux[i - 1] and
                region_flux[i] > region_flux[i + 1] and
                region_flux[i] > 0.1):
            peaks.append((i + speech_start_frame, region_flux[i]))

    # Filter peaks by minimum distance
    filtered_peaks = []
    for frame, strength in sorted(peaks, key=lambda x: -x[1]):
        too_close = False
        for existing_frame, _ in filtered_peaks:
            if abs(frame - existing_frame) < min_distance_frames:
                too_close = True
                break
        if not too_close:
            filtered_peaks.append((frame, strength))

    # Sort by time and take the N strongest
    filtered_peaks.sort(key=lambda x: x[0])

    if len(filtered_peaks) >= n_boundaries:
        # Take the N strongest peaks, then sort by time
        by_strength = sorted(filtered_peaks, key=lambda x: -x[1])[:n_boundaries]
        by_time = sorted(by_strength, key=lambda x: x[0])
        boundary_frames = [f for f, _ in by_time]
    else:
        # Not enough peaks found - interpolate between what we have
        # Use the peaks we found plus evenly-spaced fill-ins
        existing_times = [f for f, _ in filtered_peaks]
        boundary_frames = distribute_boundaries(
            existing_times, n_boundaries,
            speech_start_frame, speech_end_frame
        )

    return [frame * hop / sr for frame in boundary_frames]


def distribute_boundaries(existing_frames: list[int], n_needed: int,
                          start: int, end: int) -> list[int]:
    """Fill in missing boundaries with even spacing between existing ones."""
    if not existing_frames:
        # No peaks at all - evenly space
        step = (end - start) / (n_needed + 1)
        return [int(start + step * (i + 1)) for i in range(n_needed)]

    # Use existing peaks as anchors, fill gaps
    anchors = [start] + existing_frames + [end]
    n_extra = n_needed - len(existing_frames)

    if n_extra <= 0:
        return existing_frames[:n_needed]

    # Find the largest gaps and split them
    result = list(existing_frames)
    for _ in range(n_extra):
        # Find largest gap in current result
        all_points = sorted([start] + result + [end])
        max_gap = 0
        max_idx = 0
        for i in range(len(all_points) - 1):
            gap = all_points[i + 1] - all_points[i]
            if gap > max_gap:
                max_gap = gap
                max_idx = i
        # Split it
        mid = (all_points[max_idx] + all_points[max_idx + 1]) // 2
        result.append(mid)

    result.sort()
    return result[:n_needed]


def align_word(word: str, phoneme_map: list[dict], debug: bool = False) -> dict | None:
    """Generate timing data for a single word from audio analysis."""
    audio_path = AUDIO_DIR / word / "audio.m4a"
    if not audio_path.exists():
        print(f"  Skipping {word} (no audio file)")
        return None

    sr = 16000
    hop = 160  # 10ms hop
    frame_len = 512  # 32ms frame

    samples = load_audio_pcm(audio_path, sr=sr)
    energy = compute_energy(samples, sr=sr, frame_len=frame_len, hop=hop)
    flux = compute_spectral_flux(samples, sr=sr, frame_len=frame_len, hop=hop)

    # Find speech region
    speech_start, speech_end = find_speech_region(energy, hop=hop, sr=sr)
    start_frame = int(speech_start * sr / hop)
    end_frame = int(speech_end * sr / hop)

    n_phonemes = len(phoneme_map)
    n_boundaries = n_phonemes - 1

    # Find boundaries
    boundaries_sec = find_phoneme_boundaries(
        flux, energy, n_boundaries,
        start_frame, end_frame, hop=hop, sr=sr
    )

    # Build timing entries with minimum duration enforcement
    MIN_PHONEME_MS = 40  # 40ms minimum per phoneme
    times = [speech_start] + boundaries_sec + [speech_end]

    # Enforce minimum duration: shift boundaries away from too-short segments
    for _pass in range(3):  # iterate to resolve cascading adjustments
        for i in range(len(times) - 1):
            dur = times[i + 1] - times[i]
            if dur < MIN_PHONEME_MS / 1000:
                deficit = MIN_PHONEME_MS / 1000 - dur
                # Steal from the longer neighbor
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
        print(f"\n  {word}: speech region {speech_start:.3f}s - {speech_end:.3f}s")
        print(f"  Audio duration: {len(samples)/sr:.3f}s")
        print(f"  Boundaries: {[f'{b:.3f}' for b in boundaries_sec]}")
        for p in phonemes:
            dur = p['end'] - p['start']
            print(f"    {p['phoneme']:4s} [{p['start']:.3f} - {p['end']:.3f}] ({dur*1000:.0f}ms) → letters {p['letters']}")

    return {
        "word": word,
        "phonemes": phonemes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", help="Process only this word")
    parser.add_argument("--debug", nargs="?", const="all", help="Show detailed output (optionally for a specific word)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files, just show what would change")
    args = parser.parse_args()

    with open(MAPPINGS_PATH) as f:
        letter_mappings = json.load(f)

    words_to_process = {}
    if args.word:
        if args.word in letter_mappings:
            words_to_process[args.word] = letter_mappings[args.word]
        else:
            print(f"Word '{args.word}' not found in phoneme mappings")
            sys.exit(1)
    else:
        words_to_process = letter_mappings

    debug_word = args.debug

    generated = 0
    changed = 0

    for word, phoneme_map in words_to_process.items():
        do_debug = debug_word in ("all", word)
        result = align_word(word, phoneme_map, debug=do_debug)
        if result is None:
            continue

        output_path = AUDIO_DIR / word / "timing.json"

        # Check if timing changed
        if output_path.exists():
            with open(output_path) as f:
                old = json.load(f)
            if old == result:
                if do_debug:
                    print(f"  {word}: unchanged")
                continue

        if args.dry_run:
            print(f"  Would update: {word}")
            if do_debug:
                # Show diff
                with open(output_path) as f:
                    old = json.load(f)
                for old_p, new_p in zip(old["phonemes"], result["phonemes"]):
                    if old_p != new_p:
                        print(f"    {old_p['phoneme']}: {old_p['start']:.3f}-{old_p['end']:.3f} → {new_p['start']:.3f}-{new_p['end']:.3f}")
        else:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

        changed += 1
        generated += 1

        if not do_debug:
            print(f"  {word}: {len(phoneme_map)} phonemes aligned")

    print(f"\nProcessed {generated} words, {changed} updated")


if __name__ == "__main__":
    main()
