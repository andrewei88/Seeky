# Object Learning App — Design Spec

An iOS camera app for children ages 2-4 that identifies real-world objects and teaches the child the word through synchronized voice pronunciation with letter highlighting.

## Core Interaction

1. Child opens the app → sees a full-screen camera feed (nothing else on screen)
2. Child taps any object they see
3. If the object is confidently recognized:
   - Background dims, object glows with its pixel-perfect silhouette
   - The word appears in large lowercase letters (80pt, SF Rounded)
   - A pre-recorded voice speaks the word slowly and clearly
   - Letters highlight gold in sync with pronunciation (e.g., "ph" in "elephant" highlights together because they make one sound)
   - Child taps anywhere to return to camera
4. If the object is not confidently recognized → nothing happens. No error states exist.

## Target

- **Age range:** 2-4 years
- **Language:** English only
- **Interaction model:** Tap to learn (child-driven exploration)
- **Progress tracking:** None. Pure exploration tool.
- **Platform:** iOS 17+, iPhone only (requires physical device — segmentation APIs unavailable on simulator)

## Architecture

Five layers, each with a single responsibility:

```
┌─────────────────────────────────────────────┐
│              UI Layer (SwiftUI)              │
│   Camera preview, overlays, word display     │
├─────────────────────────────────────────────┤
│           Interaction Layer                  │
│   Tap handling, object selection, state      │
├─────────────────────────────────────────────┤
│          Detection Layer                     │
│   Segmentation (continuous) +                │
│   Classification (on-tap only)               │
├─────────────────────────────────────────────┤
│           Speech Layer                       │
│   Audio playback + letter sync animation     │
├─────────────────────────────────────────────┤
│          Data Layer                          │
│   Vocabulary, label mappings, audio files,   │
│   phoneme timing data                        │
└─────────────────────────────────────────────┘
```

**Key design principle:** Classification only runs on the single tapped object, not continuously. This maximizes compute budget for accuracy on the one object that matters.

## Detection Layer

### Subsystem A: Continuous Segmentation

- Runs `VNGenerateForegroundInstanceMaskRequest` (iOS 17+) on every 3rd camera frame (~10fps segmentation on 30fps feed)
- Returns `VNInstanceMaskObservation` with `allInstances` IndexSet
- Masks are NOT visualized — the camera feed is pure, no outlines
- Masks are tracked across frames by position/size overlap
- An instance must be stable for ~0.5 seconds before it becomes tappable (prevents phantom objects from camera shake)

### Subsystem B: On-Tap Classification

When child taps a stable instance:

1. **Hit test** — Convert tap point to image coordinates, find which instance mask contains it
2. **Crop** — Extract bounding region of that instance from the current full-resolution frame
3. **Dual classification in parallel:**
   - `VNClassifyImageRequest` (Apple's built-in, 1,303 categories) → returns fine-grained label (e.g., "golden retriever", confidence 0.92)
   - MobileCLIP image encoder (CoreML, ~30MB, 3-15ms on iPhone 12+) → compares crop against pre-computed text embeddings for all ~120 child vocabulary words → returns (e.g., "dog", similarity 0.87)
4. **Label mapping** — VNClassify's "golden retriever" is looked up in `label_mappings.json` → "dog"
5. **Consensus gate:**
   - Both agree + high confidence → SHOW
   - Disagree → REJECT
   - Either below confidence threshold → REJECT
   - VNClassify label has no mapping in vocabulary → REJECT

### Confidence Thresholds

- VNClassify: top-1 confidence ≥ 0.70, AND top-1 ≥ 1.5× top-2
- MobileCLIP: top-1 similarity ≥ 0.75, AND top-1 ≥ 1.3× top-2
- Both must resolve to the same child vocabulary word

These thresholds are deliberately conservative. Better to miss objects than mislabel them. Will require tuning during development with real-world testing.

### Label Mapping Table

`label_mappings.json` maps every relevant VNClassify label to a child vocabulary word:

```json
{
  "golden retriever": "dog",
  "labrador": "dog",
  "sedan": "car",
  "desk lamp": "lamp",
  "espresso": null
}
```

- `null` entries are explicitly rejected
- Any VNClassify label not in this file is also rejected by default
- This is a whitelist, not a blacklist — only mapped labels can produce output
- This also acts as a content filter (nothing inappropriate can appear)

## UI Layer

### Design Principles

- **Zero chrome:** No navigation bars, buttons, menus. The camera IS the app.
- **Lowercase only:** All words displayed in lowercase. Children encounter lowercase far more in books and daily life.
- **Large everything:** 80pt+ word text. Tap targets are entire object silhouettes. No precision required from small hands.
- **Warm colors:** Gold/yellow highlighting against dimmed backgrounds. High contrast, visually inviting.
- **Gentle feedback:** Soft haptic on tap. 0.3s ease animations. No jarring transitions.
- **No error states:** If detection fails, nothing happens. Child naturally taps something else.

### App States

**Exploring (default):** Pure full-screen camera. Nothing else on screen. Segmentation runs invisibly in background.

**First launch only:** A gentle pulsing hand-tap icon (👆) with the single word "tap" appears over the camera. Disappears permanently after the child's first successful object interaction. Stored as `hasCompletedFirstTap` in UserDefaults.

**Learning:** Background dims (0.6 opacity black overlay). Tapped object glows with its silhouette mask (gold glow, drop shadow). Word appears centered in large lowercase letters. Voice speaks. Letters animate. Child taps anywhere to return to exploring.

### Object Highlighting

Uses the actual instance mask from Vision framework — not a bounding box:

1. Generate a mask image for the tapped instance via `generateScaledMaskForImage(forInstances:from:)`
2. Apply the mask as an overlay: everything outside the mask gets dimmed (black, 0.6 opacity)
3. Along the mask edge, apply a gold glow effect (drop shadow / bloom)
4. The object appears to "lift" off the dimmed background — the child sees exactly what a "car" is, not a rectangle containing a car

### Word Display

- Font: SF Rounded, 80pt, bold weight
- Letter spacing: 16pt (generous spacing so each letter is distinct)
- Position: centered horizontally, upper-center of screen (above the object, not covering it)
- Color states:
  - **Not yet spoken:** dim white (rgba 255,255,255, 0.3)
  - **Currently speaking:** bright gold (#FFD93D) with glow shadow
  - **Already spoken:** softer gold (rgba 255,217,61, 0.45), fades over 0.15s

### Animations

- Dim/glow transition: 0.3s ease-in-out
- Letter state transitions: 0.15s ease-in-out
- After word completes: all letters glow fully for 0.5s
- Haptic: `UIImpactFeedbackGenerator` with `.light` style on tap

## Speech Layer

### Audio Playback

- Pre-recorded audio files bundled in app (one `.m4a` per word)
- Played via `AVAudioPlayer`
- A `CADisplayLink` (fires every screen refresh, ~60fps) checks `player.currentTime` against timing data
- At each frame: determine which phoneme is currently sounding → highlight corresponding letter(s)

**Why CADisplayLink + currentTime (not timers):**
- Timers can drift and aren't tied to actual playback position
- `currentTime` is ground truth of where audio actually is
- If audio hiccups or system lags, highlight stays correct because it checks actual position

### Letter Synchronization Timing Data

Each word has a `timing.json` file generated at build time:

```json
{
  "word": "elephant",
  "phonemes": [
    { "phoneme": "EH", "letters": [0],    "start": 0.00, "end": 0.28 },
    { "phoneme": "L",  "letters": [1],    "start": 0.28, "end": 0.52 },
    { "phoneme": "AH", "letters": [2],    "start": 0.52, "end": 0.78 },
    { "phoneme": "F",  "letters": [3, 4], "start": 0.78, "end": 1.10 },
    { "phoneme": "AH", "letters": [5],    "start": 1.10, "end": 1.35 },
    { "phoneme": "N",  "letters": [6],    "start": 1.35, "end": 1.62 },
    { "phoneme": "T",  "letters": [7],    "start": 1.62, "end": 1.85 }
  ]
}
```

Note: `"letters": [3, 4]` means indices 3 and 4 ("p" and "h") highlight together because "ph" makes one sound. This naturally teaches phonics.

### Audio Generation Pipeline (Build-Time)

**Voice provider:** ElevenLabs

- Free tier (10,000 characters/month) is sufficient for development and iteration — our full vocabulary is ~720 characters
- For longer words (3+ syllables), generate at 0.7x speed (ElevenLabs minimum) and apply gentle time-stretching in post-processing if needed
- **Commercial usage caveat:** The free tier does NOT include commercial usage rights. To ship on the App Store, the Starter plan ($5/month) is required for a commercial license. Generate all final production audio in one month and cancel — total cost $5.

**Pipeline:**

1. Generate audio for each word using ElevenLabs at 0.7x speaking rate, warm friendly voice
2. For words that need to be slower, apply time-stretching via librosa/Rubberband (pitch-preserving) as post-processing
3. Run each final audio file through Montreal Forced Aligner (v3.3.9+, sub-20ms phoneme boundary accuracy) → outputs TextGrid files
4. Python build script converts TextGrid → JSON timing format, applying hand-verified phoneme-to-letter mappings from CMU Pronouncing Dictionary (134,000+ words)
5. All ~120 phoneme-to-letter mappings are manually verified for accuracy (one-time effort)
6. Audio files + timing JSON bundled into app

## Data Layer

### Vocabulary (~120 words)

| Category | Words |
|---|---|
| Animals | dog, cat, bird, fish, horse, cow, pig, duck, rabbit, bear, elephant, lion, frog, butterfly, chicken, turtle, monkey |
| Food | apple, banana, orange, bread, egg, cheese, pizza, cake, cookie |
| Home | chair, table, bed, couch, lamp, door, window, pillow, blanket, towel, mirror, shelf, clock, fan |
| Kitchen | cup, bowl, plate, spoon, fork, knife, bottle, pot, pan, glass |
| Outdoors | car, bus, truck, tree, flower, grass, rock, leaf, fence, bench |
| Clothing | shoe, hat, shirt, pants, sock, jacket, glasses, bag |
| Bathroom | toothbrush, soap, toilet, sink, bathtub |
| Electronics | phone, tv, laptop, keyboard, remote |
| Toys | ball, teddy bear, block, doll |
| School | book, pen, pencil, paper, scissors, crayon |
| Body | hand, foot, face, eye, nose, ear |
| Other | key, light, box, umbrella, star, moon, sun, cloud, rain |

Every word is: a concrete physical noun, generic (never brand-specific), pronounceable by a 2-4 year old, common in daily life.

### MobileCLIP Text Embeddings

Pre-computed at build time by running each word through MobileCLIP's text encoder as `"a photo of a {word}"`. Stored as a binary file of embedding vectors. At runtime, only the image encoder runs — comparing the tapped object's image embedding against pre-computed text embeddings via cosine similarity (single matrix multiplication).

### App Bundle

| Component | Size |
|---|---|
| MobileCLIP image encoder (CoreML) | ~30MB |
| Audio files (120 × ~50KB) | ~6MB |
| Timing JSON files | <100KB |
| Label mappings + vocabulary | <50KB |
| Text embeddings binary | <1MB |
| **Total** | **~40MB** |

### File Structure

```
vocabulary/
├── car/
│   ├── audio.m4a
│   └── timing.json
├── dog/
│   ├── audio.m4a
│   └── timing.json
├── elephant/
│   ├── audio.m4a
│   └── timing.json
...
├── vocabulary.json
└── label_mappings.json
```

## Requirements

- iOS 17+ (for `VNGenerateForegroundInstanceMaskRequest`)
- iPhone with camera (cannot run on simulator for segmentation)
- No internet connection required — fully offline
- No accounts, no tracking, no data collection — COPPA-friendly by design
- Camera permission required (requested on first launch)

## Non-Requirements

- No iPad support (iPhone only, for now)
- No multilingual support (English only)
- No progress tracking or gamification
- No settings screen
- No parental controls (the app is safe by design — only curated words can appear)
