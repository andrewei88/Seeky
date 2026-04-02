#!/usr/bin/env python3
"""Convert MFA TextGrid output + phoneme-to-letter mappings into timing.json files."""

import json
from pathlib import Path

try:
    import textgrid
except ImportError:
    print("Install textgrid: pip install textgrid")
    raise

AUDIO_DIR = Path(__file__).parent.parent / "Seeky" / "Resources" / "Vocabulary"
MAPPINGS_PATH = Path(__file__).parent / "phoneme_to_letter_mappings.json"

def main():
    with open(MAPPINGS_PATH) as f:
        letter_mappings = json.load(f)

    for word, phoneme_map in letter_mappings.items():
        textgrid_path = AUDIO_DIR / word / "aligned.TextGrid"
        if not textgrid_path.exists():
            print(f"  Skipping {word} (no TextGrid)")
            continue

        tg = textgrid.TextGrid.fromFile(str(textgrid_path))

        # Find the phones tier
        phones_tier = None
        for tier in tg:
            if tier.name == "phones":
                phones_tier = tier
                break

        if phones_tier is None:
            print(f"  No phones tier for {word}")
            continue

        # Filter out silence/empty intervals
        phone_intervals = [
            iv for iv in phones_tier
            if iv.mark and iv.mark.strip() not in ("", "sil", "sp", "spn")
        ]

        # Match phonemes from MFA to our mapping
        timing_phonemes = []
        mapping_index = 0

        for interval in phone_intervals:
            if mapping_index >= len(phoneme_map):
                break

            expected = phoneme_map[mapping_index]
            # Strip stress markers from MFA output (e.g., "AO1" -> "AO")
            mfa_phoneme = "".join(c for c in interval.mark if not c.isdigit())

            if mfa_phoneme == expected["phoneme"]:
                timing_phonemes.append({
                    "phoneme": expected["phoneme"],
                    "letters": expected["letters"],
                    "start": round(interval.minTime, 3),
                    "end": round(interval.maxTime, 3),
                })
                mapping_index += 1
            else:
                print(f"  Warning: {word} phoneme mismatch at index {mapping_index}: "
                      f"expected {expected['phoneme']}, got {mfa_phoneme}")

        timing_data = {
            "word": word,
            "phonemes": timing_phonemes,
        }

        output_path = AUDIO_DIR / word / "timing.json"
        with open(output_path, "w") as f:
            json.dump(timing_data, f, indent=2)

        print(f"  Built timing: {word} ({len(timing_phonemes)} phonemes)")

    print("\nTiming data generation complete")

if __name__ == "__main__":
    main()
