#!/usr/bin/env python3
"""Collect remaining missing word images using icrawler (Bing).

Handles the 12 words that bing_image_downloader couldn't finish.
"""

import os
import sys
from pathlib import Path

from icrawler.builtin import BingImageCrawler
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "arya_training" / "all"
TMP_DIR = PROJECT_ROOT / "data" / "tmp_icrawl"
TARGET = 500
CROP_SIZE = (256, 256)
MIN_PX = 100

WORDS_AND_QUERIES = {
    "grass":    "green grass lawn close up",
    "key":      "metal door key",
    "leaf":     "green tree leaf close up",
    "moon":     "full moon night sky",
    "paper":    "white sheet of paper",
    "pencil":   "yellow writing pencil",
    "rain":     "rain drops falling",
    "rock":     "rock stone pebble",
    "soap":     "bar of soap bathroom",
    "speaker":  "bluetooth wireless speaker",
    "star":     "star night sky",
    "sun":      "sun bright sky sunrise",
}


def process_downloads(word, tmp_word_dir, word_dir, existing_count):
    """Resize and save downloaded images."""
    saved = 0
    for img_path in sorted(tmp_word_dir.iterdir()):
        if existing_count + saved >= TARGET:
            break
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if w < MIN_PX or h < MIN_PX:
                continue
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
            img = img.resize(CROP_SIZE, Image.LANCZOS)
            fname = f"{word}_{existing_count + saved:05d}.jpg"
            img.save(word_dir / fname, "JPEG", quality=90)
            saved += 1
        except Exception:
            continue
    return saved


def main():
    for word, query in WORDS_AND_QUERIES.items():
        word_dir = OUTPUT_DIR / word
        word_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(word_dir.glob("*.jpg")))

        if existing >= TARGET:
            print(f"[{word}] Already have {existing}, skipping")
            continue

        needed = TARGET - existing
        print(f"\n[{word}] Need {needed} more images (have {existing})...")
        sys.stdout.flush()

        # Download to temp directory
        tmp_word_dir = TMP_DIR / word
        tmp_word_dir.mkdir(parents=True, exist_ok=True)

        # Request more than needed since some will fail quality checks
        crawler = BingImageCrawler(
            storage={"root_dir": str(tmp_word_dir)},
            log_level=40,  # ERROR only
        )
        crawler.crawl(
            keyword=query,
            max_num=min(needed * 2, 1000),
            min_size=(MIN_PX, MIN_PX),
        )

        # Process and save
        saved = process_downloads(word, tmp_word_dir, word_dir, existing)
        total = existing + saved
        print(f"[{word}] Saved {saved} new images (total: {total})")
        sys.stdout.flush()

        # If still not enough, try a second query
        if total < TARGET:
            alt_queries = {
                "grass": "grass texture green field",
                "key": "house key brass silver",
                "leaf": "autumn leaf maple oak",
                "moon": "crescent moon lunar",
                "paper": "paper stack notebook blank",
                "pencil": "pencil set colored graphite",
                "rain": "rainy day window drops",
                "rock": "garden rock boulder small",
                "soap": "soap bar handmade natural",
                "speaker": "portable speaker JBL Bose",
                "star": "starry sky stars night",
                "sun": "sunset golden sun rays",
            }
            alt = alt_queries.get(word, word)
            print(f"  Trying alt query: '{alt}'...")

            # Clear tmp and re-download
            import shutil
            shutil.rmtree(tmp_word_dir, ignore_errors=True)
            tmp_word_dir.mkdir(parents=True, exist_ok=True)

            crawler = BingImageCrawler(
                storage={"root_dir": str(tmp_word_dir)},
                log_level=40,
            )
            crawler.crawl(
                keyword=alt,
                max_num=min((TARGET - total) * 2, 1000),
                min_size=(MIN_PX, MIN_PX),
            )
            saved2 = process_downloads(word, tmp_word_dir, word_dir, total)
            total += saved2
            print(f"  Alt query added {saved2} (total: {total})")

    # Cleanup
    import shutil
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    # Final report
    print("\n=== FINAL COUNTS ===")
    for word in sorted(WORDS_AND_QUERIES.keys()):
        word_dir = OUTPUT_DIR / word
        count = len(list(word_dir.glob("*.jpg"))) if word_dir.exists() else 0
        status = "OK" if count >= 200 else "LOW"
        print(f"  {word}: {count} [{status}]")


if __name__ == "__main__":
    main()
