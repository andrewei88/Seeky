# ARYA Project Instructions

## Development Process

### One feature at a time
- Fix or implement ONE thing, build, test on device, confirm it works, THEN move to the next.
- Never batch multiple fixes into one round. Each change can introduce new bugs that mask each other.
- If you can't test something (e.g., camera/audio/visual behavior), say so explicitly and ask the user to test on device.

### Test everything before claiming it works
- Write a unit test for every piece of logic before or immediately after changing it.
- Run `xcodebuild test` after every change, not just at the end.
- "It compiles" is NOT verification. "The tests pass" is partial verification. "It works on device" is real verification.
- If a feature can't be unit tested (audio playback, camera, visual overlays), add detailed `print()` logging so the user can report console output after testing on device.

### Don't guess at API behavior
- If you're unsure how an iOS API works (e.g., does `videoRotationAngle` rotate pixel data or just set metadata?), say so and add runtime logging to check, rather than assuming.
- Audio session categories have real consequences (`.ambient` is silenced by ringer switch). Verify behavior, don't assume.
- CoreImage coordinate systems (Y-up) differ from UIKit (Y-down). Reason through transforms carefully and log intermediate values.

### When something goes sideways, STOP
- If a fix doesn't work or introduces new problems, stop and re-plan immediately — don't keep pushing forward.
- Make every change as simple as possible. Touch only what's necessary.
- Before presenting a fix, ask: "Would a staff engineer approve this?"

### When debugging
- Read the user's logs line by line. They contain the answers.
- Don't propose a fix until you've identified the root cause.
- Planning documents and task trackers don't find bugs. Reading code and writing tests do.

### Expensive objects — create once, reuse
- `CIContext` is expensive to allocate (GPU resources, shader pipeline caching). Never create one per-call. Use `sharedCIContext` from `ImageUtils.swift`.
- Same principle applies to any heavy framework object (`MLModel`, `VNSequenceRequestHandler`, etc).

### Apply documented lessons to ALL code paths
- When CLAUDE.md documents a known gotcha (e.g., "mask pixel buffer can be Float32"), grep for every code path that touches that data type and verify the lesson is applied.
- Documenting a bug isn't enough — the fix must cover all call sites.

### Shared utilities live in `ImageUtils.swift`
- `cosineSimilarity`, `resizePixelBuffer`, and `sharedCIContext` live in `ARYA/Data/ImageUtils.swift`.
- Check there before writing image/vector math helpers. Add new shared utilities to the same file.

### Integration Verification
- After wiring a new model or pipeline, verify by checking for its specific log output in device/simulator logs before marking integration complete
- "Task complete" means the new code is RUNNING, not just compiling — check that the old code path is no longer executing
- When changing embedding dimensions (e.g., 512→1024), existing CorrectionStore data becomes invalid — warn the user to clear corrections

### Never git checkout over unstaged generated artifacts
- Before ANY `git checkout` that touches file paths, run `git status` and check for unstaged changes in those paths
- If unstaged changes exist, `git stash` or copy files to a safe location first
- For model files: always save versioned copies to `models/versions/` BEFORE any git operations
- `git checkout HEAD --` restores to last committed state, NOT to what was there before a previous checkout
- Prefer `git diff` or `git show` to inspect old versions without modifying the working tree
- **Incident (2026-03-27):** `git checkout` destroyed the retrained CoreML model that had just been converted. PyTorch checkpoint survived only because it was in a different directory.

## Project-Specific Knowledge

### Architecture
- iOS 17+, Swift 5.9, XcodeGen (`project.yml` → `ARYA.xcodeproj`)
- Run `xcodegen generate` after adding/removing files
- Five layers: Views → App (state) → Detection → Speech → Data
- Single-model classification: Custom classifier (CoreML) with confidence threshold (0.40). CorrectionStore checked first.
- VNClassifyImageRequest is NOT used. Removed from classification and environment detection. Vision framework used only for segmentation.
- Custom classifier: ARYAClassifier.mlpackage (151 classes, 12.9MB). Input: 224x224 RGB. Outputs: softmax probabilities + 1024-dim feature vector
- Backbone: FastViT-T12 (Apple, 6.7M params, 79.3% ImageNet). Val accuracy: 95.2% (top-5: 99.1%). Upgraded from MobileNetV3-Small (91.3% val). Training uses timm (`fastvit_t12`) with class-weighted loss.
- MLMultiArray on ANE outputs Float16 — always use subscript access (`array[i].floatValue`), never `dataPointer.bindMemory(to: Float.self)`
- Training pipeline: `collect_training_data.py` → `curate_training_data.py` → `train_classifier.py` → `convert_to_coreml.py` → `verify_classifier.py`
- Retraining pipeline: Parent exports captures from app → `ingest_phone_captures.py [zip]` → `train_classifier.py` → `convert_to_coreml.py`
- Learning progression: `WordProgressStore` dual-tracks explore (feeds quiz pool) and quiz (feeds mastery/spaced repetition). Mastery levels 0-4 based on consecutive quiz correct answers.
- Quiz/scavenger hunt mode: Primary feature. 5 words/session from quiz pool. Child taps matching object. Parent pencil override in both directions.
- Quiz pool: Seeded with high-confidence objects, expanded by explore mode identifications (2+ IDs, 0 corrections per word)
- Explore mode: Secondary feature, accessible from scavenger hunt screen. Free-roaming object identification.
- Parent settings: Gear icon in top-right corner. Shows quiz mode, learning progress, capture stats, export/clear actions.
- ConsensusGate.swift deleted (VN removed from classification path, March 2026). LabelMapper.swift kept for test utilities.
- MobileCLIP S0 files (CLIPEmbeddings.swift, MobileCLIPImageEncoder.mlpackage, text_embeddings.bin) are legacy — kept for fallback but no longer in the active classification path
- **Planned rebrand: ARYA -> Seeky** (scavenger hunt concept: seek + playful suffix)

### Known Device Issues (March 2025)
- **Laptop/monitor confusion**: Custom classifier splits ~0.52/0.47 when tapping screen portion. White screen content triggers it. Web val accuracy (94.2% laptop, 90.7% monitor) overstates real-world performance because web photos show distinctive hinges/keyboards.
- **Light/moon VN confusion**: FIXED by removing VN from classification. Custom correctly identifies lights (0.81-0.88).
- **Slow app startup**: 12.9MB FastViT model takes longer for CoreML to compile on first launch vs old 3.3MB model. Investigate lazy loading or INT8 (6.7MB).

### Camera & Coordinates
- **Tap-to-pixel coordinate offsets are the #1 recurring bug.** Never assume screen coordinates, device coordinates, and buffer coordinates are in the same space. They aren't. Log all three at the point of conversion and verify on device before trusting the math.
- Safe area insets cause silent offsets: a view ignoring safe area and a sibling respecting it have different coordinate origins. The 59pt status bar offset caused the glow-above-tap bug. When multiple views overlay each other, confirm they share the same coordinate space.
- Camera output has `videoRotationAngle = 90` on the data output connection
- The actual buffer orientation (portrait vs landscape) must be checked at runtime via `CVPixelBufferGetWidth/Height`
- The mask and tap coordinate transforms depend on this — DO NOT hardcode an assumption
- `AppState.bufferIsLandscape` is set from the first frame and drives coordinate mapping

### Audio
- Pre-recorded ElevenLabs `.m4a` files at `Vocabulary/{word}/audio.m4a` (155 words)
- Phoneme timing at `Vocabulary/{word}/timing.json` (generated by Montreal Forced Aligner)
- `WordSpeaker` uses `AVAudioPlayer`, NOT `AVSpeechSynthesizer`
- Audio session must be `.playback` with `.mixWithOthers` (not `.ambient`, which is silenced by ringer)
- `Vocabulary/` is a folder reference in the Xcode project (not individual file references)

### Detection
- `VNGenerateForegroundInstanceMaskRequest` for segmentation (~10fps, every 3rd frame)
- Classification: single custom FastViT-T12 model (CorrectionStore checked first, then model with 0.40 confidence threshold)
- MobileCLIP S0 files are legacy — kept for fallback but not in active classification path
- `InstanceTracker` has a 1.5s grace period — instances survive brief segmentation dropouts
- Tap hit testing uses padded bounding boxes (4% padding)
- Null-mapped labels (document, screenshot, machine, etc.) are skipped when ranking classification results

### Mask Overlay Rendering (HARD-WON LESSONS)
- **CIContext.createCGImage DROPS alpha.** CGImages created from CIImage via `createCGImage(_:from:)` do NOT preserve the alpha channel. Every attempt to use CIColorMatrix to set alpha, then render to CGImage, produces fully opaque images. This breaks both dim and glow layers.
- **The ONLY reliable approach**: Use `CGContext` with explicit `CGImageAlphaInfo.premultipliedLast` to construct RGBA pixel data manually, then call `makeImage()`. This gives you a CGImage with correct alpha.
- **Do NOT use CALayer.mask with CIImage-derived CGImages** — CALayer.mask uses the mask layer's alpha channel, and CIImage-derived CGImages have alpha=1 everywhere.
- **Performance: edge detection on full-res mask (1080×1920) is too slow** (~3.6 seconds). Downsample the mask to half resolution (540×960) before distance transforms. Use `resizeAspectFill` on the glow layer to upscale — the slight blur actually improves the glow appearance.
- **Mask pixel format**: VNInstanceMaskObservation.generateScaledMaskForImage returns a single-channel CVPixelBuffer that can be **either UInt8 OR Float32** (`kCVPixelFormatType_OneComponent32Float`, format code `0x4c303066`). Must check `CVPixelBufferGetPixelFormatType` and read accordingly: Float32 needs 4-byte stride and 0–1 → 0–255 conversion. Reading Float32 as UInt8 produces garbage data (this caused the "random glow placement" bug).
- **Glow needs BOTH inner and outer**: A glow only on object pixels (inner) looks like a thin line. Must compute two distance transforms — one from object→background (inner glow) and one from background→object (outer glow) — to create a visible halo effect.
- **Box blur produces the best glow**: Create a thin binary edge strip (2px at boundary), then apply 2 passes of separable box blur (radius 12 at half-res). This approximates Gaussian blur and creates a smooth, professional glow. Boost intensity 4x after blur so the core stays bright. Much better than manual quadratic/exponential falloff.
- **Portrait/landscape**: Check buffer dimensions at runtime. If landscape (W > H), rotate coordinates when reading pixels: portrait(px, py) = landscape(py, H-1-px).
- **Vision objects are NOT thread-safe**: `VNInstanceMaskObservation.generateScaledMaskForImage(from:)` and `VNImageRequestHandler` must be used on the main thread. Calling them from a background thread causes silent failures. The pattern: generate mask + read pixel data on main thread, dispatch pure computation (distance transforms, blur, CGImage creation) to background.
- **DO NOT live-track the mask**: Trying to re-render a pixel-level mask at ~3fps while the camera runs at 30fps looks terrible — laggy, imprecise, jittery. Instead, **freeze the camera frame at tap time**: capture `CVPixelBuffer` → `UIImage`, display it on top of the live camera feed, and render the mask overlay once on the frozen frame. This is what Google Lens, Apple Visual Look Up, etc. all do. Perfect alignment, no lag.

### Audio Generation
- ElevenLabs voice: Jessica (`cgSgspJ2msm6clMCkdW9`) — "Playful, Bright, Warm"
- Speed 0.7x is ElevenLabs minimum (valid range: 0.7–1.2)
- Free tier cannot use library voices (like "Rachel") via API — use premade voices only
- Output: mp3_44100_128, converted to .m4a via ffmpeg
- Phoneme timing is generated by `scripts/align_timing_from_energy.py` (spectral flux onset detection, no MFA required)
- MFA-based alignment also available via `scripts/align_audio.py` + `scripts/build_timing_data.py` (requires conda install of MFA)

### Testing
- Tests in `ARYATests/` — run with: `xcodebuild test -scheme ARYA -destination 'platform=iOS Simulator,name=Test iPhone' -only-testing:ARYATests`
- `BundleResourceTests` verifies all 155 audio + timing files are accessible in the built bundle (device-only)
- `ClassificationEngineTests` verifies null-mapped label skipping, threshold logic, and confidence aggregation
- `ConsensusGateTests` verifies dual-model consensus logic (agreement, CLIP override, thresholds)
- `CorrectionStoreTests` covers add/lookup, threshold boundaries (0.70), centroid matching, undo, persistence, multi-word disambiguation
- `InstanceTrackerTests` covers grace period, tap padding, IoU matching
- `LetterHighlighterTests` covers letter state transitions, multi-letter phonemes, space handling
- `CropLogicTests` tests tap-centered crop and device-to-buffer coordinate conversion
- `TrainingCaptureTests` covers stats, export zip creation, clear, underscore word handling
- `WordProgressStoreTests` covers explore/quiz dual tracking, quiz eligibility, mastery progression, word selection, persistence
- `ClassificationIntegrationTests` runs VN on real images (device-only, downloads from Unsplash)
- Tests that require Vision framework or bundle resources only pass on device, not simulator
