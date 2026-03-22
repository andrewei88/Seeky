# ARYA: Teaching Toddlers Words Through the Camera

## What Is This Thing?

ARYA is an iOS camera app for kids aged 2-4. The whole idea is stupidly simple and that's what makes it good: a toddler opens the app, sees the camera, taps on something they see in the real world, and the app says the word out loud while lighting up each letter in sync with the pronunciation. That's it. No accounts, no ads, no progress bars, no settings screens. Just a camera and a child's curiosity.

Think of it like a patient teacher who never gets tired, never mispronounces a word, and never shows a kid the wrong label. That last part -- never showing the wrong label -- turned out to be the hardest engineering problem in the whole project.

---

## The Five-Layer Architecture (And Why It Matters)

The app is built like a layer cake, and each layer only talks to the ones directly above and below it:

```
 UI Layer          — What the child sees (SwiftUI)
 Interaction Layer — What happens when they tap (AppState)
 Detection Layer   — What the camera sees (Vision + CoreML)
 Speech Layer      — What the app says (AVAudioPlayer + CADisplayLink)
 Data Layer        — What the app knows (vocabulary, embeddings, timing data)
```

Why five layers instead of just writing everything in one ViewController like a cowboy? Because each layer has a fundamentally different job, and keeping them separate means you can change one without breaking the others. The detection layer doesn't know or care about SwiftUI. The speech layer has no idea there's a camera. AppState is the conductor of the orchestra -- it's the only thing that talks to everybody.

This is a pattern worth internalizing: **when you find yourself writing a class that does camera work AND plays audio AND manages UI state, you've gone wrong somewhere.** Split it up. Your future self will thank you.

---

## The UI Layer: Zero Chrome, Maximum Trust

Open `/ARYA/Views/ContentView.swift` and you'll notice something unusual: there's almost nothing there. No navigation bars. No buttons. No menus. The camera IS the interface. This is a deliberate design philosophy called "zero chrome," and it's perfect for a 2-year-old who can't read a menu anyway.

The entire view hierarchy is a ZStack with three possible layers:

1. **CameraPreviewView** -- always visible, full screen, wrapping an AVCaptureVideoPreviewLayer
2. **LearningOverlayView** -- appears when a word is being taught (dims the background, highlights the object, shows the word)
3. **OnboardingHintView** -- a pulsing tap icon shown exactly once on first launch, then gone forever

When the child taps an object and it's recognized, the `LearningOverlayView` takes over. This is where the magic happens visually. Look at `MaskOverlayView.swift` -- it uses the actual pixel-perfect silhouette mask from Vision framework (not a boring rectangle) to create a "spotlight" effect. Everything outside the object dims to 60% black, and the object itself gets a warm gold glow using CIGaussianBlur on the mask edges. The child literally sees the object "lift" off the screen. It's beautiful.

The `WordDisplayView` renders each letter individually in 80pt SF Rounded Bold with 16pt spacing between letters. Each letter has its own color state (dim white for upcoming, bright gold for currently speaking, softer gold for already spoken), and transitions happen with 0.15s ease-in-out animations. The effect is like a karaoke bouncing ball, but for learning the word "elephant."

---

## The Detection Layer: Two Brains Are Better Than One

Here's where the engineering gets interesting. The detection layer has two subsystems that work completely differently, and they have to agree before anything is shown to a child.

### Subsystem A: Continuous Segmentation (The Eyes)

`SegmentationEngine.swift` runs Apple's `VNGenerateForegroundInstanceMaskRequest` on every 3rd camera frame. That's roughly 10 segmentations per second on a 30fps camera feed. It finds every distinct object in the frame and generates a pixel-perfect mask for each one.

But here's the thing -- the child never sees any of this. There are no outlines, no bounding boxes, no visual hints. The camera feed looks completely clean. The segmentation runs silently in the background like a guard dog watching but not barking.

`InstanceTracker.swift` then tracks these detected instances across frames using IoU (Intersection over Union) -- basically asking "is this the same object I saw last frame, just moved slightly?" An instance must be stable for 0.5 seconds before it becomes tappable. This prevents phantom objects from camera shake. Imagine a toddler's hand jiggling the phone -- without this stability requirement, the app would detect objects that flicker in and out of existence.

### Subsystem B: On-Tap Classification (The Brain)

Classification is the expensive part, so it only runs when the child actually taps something. This is a key architectural decision: **don't waste compute on things the user didn't ask about.** The child tapped ONE object. Classify that ONE object. Save all your GPU budget for getting that one right.

When a tap lands on a stable instance, `ClassificationEngine.swift` crops just that object from the full-resolution frame and runs TWO classifiers in parallel using Swift's `async let`:

1. **Apple's VNClassifyImageRequest** -- built into iOS, knows 1,303 categories. Very accurate but very specific. It'll say "golden retriever" when a kid just needs to hear "dog."

2. **MobileCLIP S0** (Apple's 22MB CoreML model) -- compares the cropped image against pre-computed text embeddings for all 107 child vocabulary words. It thinks in terms of "how much does this image look like 'a photo of a dog'?"

These two models see the world differently. VNClassify is a taxonomist -- precise, specific, sometimes pedantic ("golden retriever" when the kid needs "dog"). MobileCLIP is more like a vibes-based thinker -- it gets the gist. VNClassify also has blind spots where it returns unhelpful labels like "document" or "screenshot" for screens, returning nothing useful. In those cases, CLIP steps in as the sole classifier. When both models have opinions, CLIP is trusted as the primary signal because it classifies directly against our vocabulary rather than through a 1,303-category intermediary.

### The Consensus Gate: Trust, But Verify

`ConsensusGate.swift` is the bouncer at the door. It uses a CLIP-primary strategy:

- VNClassify's raw label (like "golden retriever") gets mapped to a child word (like "dog") through `LabelMapper`. Confidence is **aggregated** across all VN labels that map to the same word (so `cup` + `mug` both contribute to the "cup" score).
- If both models agree, accept with a lower bar (CLIP similarity ≥ 0.20)
- If they disagree, trust CLIP if its similarity meets threshold and has sufficient margin over the second choice
- If VN has NO mapped word (common when it returns "document" or "screenshot" for screens), CLIP classifies solo
- If neither model is confident enough, the tap is silently rejected

The thresholds are deliberately conservative. **It is better to miss 20 real objects than to tell a 2-year-old that a cat is a dog.** This is a fundamental design philosophy that should guide every children's app: when in doubt, do nothing. Kids don't get frustrated by silence. They just tap something else. But a wrong label could teach them an incorrect word they'll repeat for months.

The `LabelMapper` acts as a whitelist. Only labels that explicitly map to a child vocabulary word can produce output. "Espresso" maps to `null` (rejected). Any VNClassify label not in the mapping file is also rejected by default. This doubles as a content filter -- nothing inappropriate can ever appear on screen because only curated words exist in the vocabulary.

### The CameraManager: Plumbing That Matters

`CameraManager.swift` is straightforward AVFoundation code -- set up an `AVCaptureSession`, configure the back camera at 1920x1080, deliver frames via delegate. But two details matter:

1. **Frame skipping**: `frameCount % 3 == 0` means only every 3rd frame gets processed. Running segmentation at 30fps would cook the battery and generate heat a child would feel. At 10fps, it's plenty responsive and thermally sustainable.

2. **`alwaysDiscardsLateVideoFrames = true`**: If segmentation takes longer than expected on one frame, don't queue up a backlog. Just drop the frame and move on. Latency matters more than completeness for real-time camera apps.

---

## The Speech Layer: Timing Is Everything

### WordSpeaker: The Voice

`WordSpeaker.swift` plays pre-recorded audio files (one `.m4a` per word) and reports the current playback time at 60fps using `CADisplayLink`. The key insight here is that letter highlighting is driven by *actual audio position*, not by timers.

Why does this matter? Timers drift. If you set a timer for "highlight the letter 'e' at 0.28 seconds," and the system is busy processing something else, that timer might fire at 0.31 seconds. Meanwhile the audio might have hiccupped and only be at 0.25 seconds. Now your highlight is wrong. But `CADisplayLink` + `AVAudioPlayer.currentTime` always gives you ground truth. "Where is the audio RIGHT NOW?" Then you figure out which letter should be highlighted at that exact moment. If audio lags, highlights lag with it. They stay in sync because they're driven by the same source of truth.

This is a principle worth remembering: **whenever you need two things in sync, derive them both from a single source of truth.** Don't have two independent clocks.

### LetterHighlighter: The Choreographer

`LetterHighlighter.swift` takes the current audio time and the word's timing data and produces a `[LetterState]` array -- one state per character. The timing data knows things like "the 'ph' in 'elephant' highlights together because they make one sound." This naturally teaches phonics without the app even trying.

Here's a quick example of how the timing data works for "elephant":

```
phoneme "EH" → letter [0] (e)      from 0.00s to 0.28s
phoneme "L"  → letter [1] (l)      from 0.28s to 0.52s
phoneme "AH" → letter [2] (e)      from 0.52s to 0.78s
phoneme "F"  → letters [3,4] (ph)  from 0.78s to 1.10s  ← two letters, one sound!
phoneme "AH" → letter [5] (a)      from 1.10s to 1.35s
phoneme "N"  → letter [6] (n)      from 1.35s to 1.62s
phoneme "T"  → letter [7] (t)      from 1.62s to 1.85s
```

At time 0.90 seconds, the highlighter would return: `[.spoken, .spoken, .spoken, .active, .active, .upcoming, .upcoming, .upcoming]`. Letters 3 and 4 ("p" and "h") are both `.active` because they're making one sound together.

---

## The Data Layer: More Work Than You'd Think

### Vocabulary

107 words, all concrete physical nouns a 2-4 year old encounters in daily life. Categories include animals, food, home items, kitchen items, outdoor things, clothing, and more. Every word was chosen to be: pronounceable by a toddler, generic (never "Tesla" -- always "car"), and physically present in typical daily life.

### CLIPEmbeddings: Pre-Computed Cleverness

`CLIPEmbeddings.swift` is where a neat optimization lives. MobileCLIP has two halves: a text encoder and an image encoder. At build time, every vocabulary word is run through the text encoder as "a photo of a {word}" and the resulting 512-dimensional embedding vectors are saved as a binary file (`text_embeddings.bin`, 214KB). At runtime, only the image encoder runs (22MB CoreML model, 1.5ms on Neural Engine). Comparing an image to 107 words is then just cosine similarity using Apple's Accelerate framework (vDSP). This is dramatically faster than running both encoders at runtime.

The image encoder expects 256x256 RGB input, so `CLIPEmbeddings` resizes the crop via CIImage before inference. The output embedding is L2-normalized before comparison. The cosine similarity computation uses `vDSP_dotpr` for the dot product and norms -- hardware-accelerated vector math that runs on the CPU's SIMD units. For 107 words with 512-dimensional embeddings, this comparison takes microseconds.

### Build-Time Audio Pipeline

This is the part that lives outside the app, in the `scripts/` directory. The pipeline goes:

1. **ElevenLabs API** generates voice recordings for each word at 0.7x speed (warm, friendly, slow enough for toddlers)
2. **Montreal Forced Aligner** (MFA v3.3.9+) analyzes each audio file and identifies exactly when each phoneme starts and ends, with sub-20ms accuracy
3. A Python script converts MFA's TextGrid output into the `timing.json` format the app expects
4. Hand-verified phoneme-to-letter mappings (from CMU Pronouncing Dictionary) ensure things like "ph" → one sound are correct

All 107 words have pre-generated audio and timing data bundled into the app. The app is fully offline -- no internet needed, ever.

---

## The Orchestration: AppState

`AppState.swift` is the conductor. It's a `@MainActor` `ObservableObject` that owns every major component:

- `CameraManager` -- delivers frames
- `SegmentationEngine` -- finds objects in frames
- `InstanceTracker` -- stabilizes detections
- `ClassificationEngine` -- identifies tapped objects (which itself owns `LabelMapper`, `CLIPEmbeddings`, and `ConsensusGate`)
- `WordSpeaker` -- plays audio
- `VocabularyStore` -- knows all the words

The state machine is dead simple: `exploring` → `classifying` → `learning` → back to `exploring`. If classification fails at any point, it silently returns to `exploring`. No error dialogs. No "sorry, we couldn't identify that." Just... nothing happens. The child taps something else.

The `handleTap` method is worth studying. It checks the mode, performs a haptic, sets mode to `.classifying`, crops the pixel buffer, runs dual classification through async/await, and either transitions to `.learning` or silently returns to `.exploring`. The entire flow is clean and linear despite being asynchronous.

---

## Technologies Used (And Why)

| Technology | Why This One |
|---|---|
| **SwiftUI** | The overlay animations (per-letter color transitions, opacity fades) are trivially declarative. UIKit would require significantly more animation management code. |
| **Vision Framework** | `VNGenerateForegroundInstanceMaskRequest` gives pixel-perfect object silhouettes for free (no ML model needed). `VNClassifyImageRequest` provides 1,303-category classification built into iOS. No downloads, no API keys. |
| **CoreML + MobileCLIP S0** | Apple's own lightweight CLIP model for on-device zero-shot classification. 22MB image encoder, 1.5ms inference on iPhone 12+ Neural Engine. Text encoder runs at build time to pre-compute 107 word embeddings (214KB); only the image encoder ships in the app. Downloaded from Apple's official HuggingFace repo (`apple/coreml-mobileclip`). |
| **AVFoundation** | The only real option for camera access on iOS. We need raw pixel buffers (not just a viewfinder) so we can run Vision requests on them. |
| **AVAudioPlayer + CADisplayLink** | `AVAudioPlayer.currentTime` gives ground-truth playback position. `CADisplayLink` fires at display refresh rate (~60fps). Together they provide frame-accurate audio-visual sync without timer drift. |
| **XcodeGen** | The `project.yml` file is 40 lines. The generated `.xcodeproj` is thousands of lines of XML. Version-controlling a YAML file instead of an Xcode project file prevents merge conflicts and makes the project definition human-readable. |
| **Accelerate (vDSP)** | Hardware-accelerated vector math for cosine similarity. When you're comparing a 512-dimensional vector against 120 others, you want SIMD, not a for-loop. |
| **ElevenLabs** | Natural-sounding TTS with speed control. The free tier (10,000 chars/month) covers the whole vocabulary (~720 chars) many times over for development. |
| **Montreal Forced Aligner** | Sub-20ms phoneme boundary accuracy. This is what makes the letter highlighting feel magical instead of janky. |

---

## Bugs We Hit and How We Fixed Them

### 1. The WordSpeaker NSObject Saga

`WordSpeaker` needs to be an `AVAudioPlayerDelegate` to know when audio finishes playing. `AVAudioPlayerDelegate` is an Objective-C protocol, which means the conforming class must inherit from `NSObject`. But we also needed it to be an `ObservableObject` to publish `currentTime` to SwiftUI.

The fix: `final class WordSpeaker: NSObject, ObservableObject`. This works because `NSObject` is a class and `ObservableObject` is a protocol. Swift allows a class to inherit from one class and conform to multiple protocols. But you have to put the class first in the declaration -- `NSObject, ObservableObject` -- not the other way around.

**Lesson:** When bridging Swift and Objective-C patterns (which happens a lot with AVFoundation and UIKit), you'll frequently need NSObject as a base class. Know when and why.

### 2. The Vocabulary Folder Reference Collision

Here's a sneaky one. XcodeGen was trying to compile the `Vocabulary/` folder contents as Swift source files. Audio files and JSON files being fed to the Swift compiler. Naturally, it exploded.

The fix is in `project.yml`:

```yaml
sources:
  - path: ARYA
    excludes:
      - "Resources/Vocabulary"      # Exclude from source compilation
  - path: ARYA/Resources/Vocabulary
    type: folder                     # Add back as a folder reference
    buildPhase: resources            # Put it in the resources build phase
```

You have to EXCLUDE the Vocabulary directory from the main source path, then re-add it explicitly as a folder reference with `type: folder` and `buildPhase: resources`. If you just mark it as a folder without excluding it first, XcodeGen sees it twice and complains. If you exclude it without re-adding it, the audio files don't get bundled.

**Lesson:** XcodeGen's `type: folder` creates a folder reference (blue folder in Xcode) instead of a group (yellow folder). Folder references include ALL contents at build time without listing each file individually. This is essential when you have 103 subdirectories each containing audio and JSON files -- you don't want to enumerate them all in your project config.

### 3. ElevenLabs Free Tier Limitations

The free tier generates great audio for development, but doesn't include commercial usage rights. If you ship to the App Store, you technically need the Starter plan ($5/month). The clever workaround: subscribe for one month, regenerate all 103 audio files with the commercial license, cancel the subscription. Total cost: $5. All audio lives in the app bundle forever.

**Lesson:** Always check the licensing terms of third-party services BEFORE building your pipeline around them. We got lucky that ElevenLabs' paid tier is cheap and you can generate everything in one batch. Some services require ongoing subscriptions for any commercial use of previously generated content.

### 4. CameraManager Delegate Threading

`CameraManager` delivers frames on its `outputQueue` (a background DispatchQueue), but `AppState` needs to update `@Published` properties on the main thread. The solution is the `nonisolated` keyword on the delegate method in AppState, which allows it to be called from any thread, combined with `Task { @MainActor in ... }` to bounce the state updates back to the main actor.

```swift
extension AppState: CameraManagerDelegate {
    nonisolated func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        let result = segmentationEngine.segment(pixelBuffer: pixelBuffer) // runs on background thread
        Task { @MainActor in
            instanceTracker.update(instances: result.instances, timestamp: timeSeconds) // updates on main
        }
    }
}
```

**Lesson:** In Swift concurrency, be deliberate about what runs where. Heavy work (segmentation) should happen off the main thread. State updates (`@Published` properties) must happen on the main thread. The `nonisolated` keyword and `@MainActor` give you fine-grained control.

---

## Best Practices and How Good Engineers Think

### 1. "No Error States" Philosophy

This app has zero error dialogs. Zero loading spinners. Zero "something went wrong" messages. If classification fails, nothing happens. If the camera can't start, the screen is just black. If audio files are missing, the word appears without sound.

This isn't laziness -- it's intentional design for a 2-year-old user. A toddler doesn't understand "classification confidence below threshold." They just tap something else. Every possible failure mode silently degrades to a reasonable state. This is called **graceful degradation** and it's especially important in apps for users who can't read error messages.

### 2. Conservative Thresholds for Children

The thresholds (CLIP similarity >= 0.20 with margin >= 1.05x; VN-only fallback at 0.03 confidence) are calibrated to reject ambiguous results while still recognizing most common objects. We'd rather miss some valid objects than mislabel a single one. In an adult app, you might show a "did you mean...?" prompt. A 2-year-old can't evaluate whether a suggestion is correct. Whatever the app says, they'll believe it.

This is a broader principle: **your error tolerance should match your user's ability to detect and recover from errors.** Programmers can handle error messages. Adults can evaluate suggestions. Toddlers cannot.

### 3. Frame Skipping for Battery Life

Processing every 3rd camera frame (`frameCount % 3 == 0`) is a simple optimization with a huge impact. Segmentation is expensive. Running it at 30fps would drain the battery in minutes and make the phone hot enough that a parent would take it away. At 10fps, segmentation is still responsive (objects stabilize in 0.5 seconds anyway) but power consumption drops dramatically.

**Lesson:** Before optimizing the algorithm, ask if you can just run it less often. Often the cheapest optimization is calling the expensive function fewer times.

### 4. The Whitelist Approach

The label mapper uses a whitelist, not a blacklist. Only explicitly mapped labels produce output. This means:
- You can never accidentally show an inappropriate word
- New VNClassify categories in future iOS versions are automatically excluded
- The vocabulary is exactly what you curated, nothing more

**Lesson:** For safety-critical features (and teaching words to children is safety-critical in its own way), prefer whitelists over blacklists. A blacklist requires you to think of everything bad. A whitelist only requires you to think of everything good. The second list is shorter and more auditable.

### 5. Single Source of Truth for Timing

The letter-highlighting system reads audio position from `AVAudioPlayer.currentTime` via `CADisplayLink`, not from independent timers. This means highlights can never drift out of sync with audio, even if the system is under load. Both the audio position and the visual highlight derive from the same clock.

### 6. IoU for Object Tracking

Instead of relying on Vision's instance IDs across frames (which aren't guaranteed stable), InstanceTracker uses Intersection over Union to match objects between frames. Two bounding boxes with IoU > 0.3 are considered the same object. This is robust to small movements and size changes, which happen constantly when a toddler holds a phone.

---

## Potential Pitfalls (And How to Avoid Them)

### Simulator Won't Work

`VNGenerateForegroundInstanceMaskRequest` requires a physical device running iOS 17+. If you try to run on the simulator, segmentation silently returns empty results. There's no crash, no error -- just no objects detected. This can be very confusing if you don't know about it. Always test on a real device.

### CoreML Model Compatibility

MobileCLIP's CoreML model needs to match the input dimensions and output feature names your code expects. The S0 model takes 256x256 RGB images as `"image"` input and returns 512-dim embeddings as `"final_emb_1"` output. If you swap in a different CLIP variant (S1, S2, B), you'll need to verify these names match. A model mismatch will return nil from prediction silently. The `CLIPEmbeddings.resizePixelBuffer` helper handles resizing the crop to 256x256 automatically.

### Memory Pressure from CVPixelBuffers

`CVPixelBuffer` objects from the camera can be large (1920x1080 BGRA = ~8MB each). The frame-skipping and `alwaysDiscardsLateVideoFrames = true` help, but be careful about retaining pixel buffers longer than necessary. The segmentation result stores a reference to the pixel buffer for mask generation, so it stays alive as long as `latestSegmentation` is set. If you ever add code that keeps old segmentation results around, you could accumulate significant memory pressure.

### Audio Session Configuration

`WordSpeaker` sets the audio session category to `.playback` before each play. If you don't do this, the audio might play at reduced volume or not at all when the phone is in silent mode. The `.playback` category means "this audio is the primary purpose of the app" -- which is true, since you're teaching a child a word.

### CIContext Reuse

`MaskUIView` creates a single `CIContext` and reuses it for all mask rendering. Creating a new `CIContext` for every frame would be extremely expensive. If you ever refactor the mask rendering, make sure the context is created once and reused. This is a common Core Image performance trap.

### Timing Data Must Match Audio

If you regenerate audio files without regenerating timing data (or vice versa), the letter highlights will be out of sync. The build pipeline generates both together, but if you manually replace an audio file, remember to re-run MFA alignment and rebuild the timing JSON.

### The "Phase Problem" in Phoneme Mapping

Some English words have ambiguous phoneme-to-letter mappings. Consider "knight" -- is the "n" sound mapped to the letter "k" (which is silent) or "n"? The CMU Pronouncing Dictionary handles pronunciation, but the *mapping back to letters* requires manual verification. All 107 words have been hand-verified, but if you add new words, expect to spend a few minutes per word checking these mappings.

---

## How the Pieces Connect (The Full Flow)

Let's trace what happens when a 3-year-old named Arya taps a dog on screen:

1. `CameraManager` is delivering 30fps frames. Every 3rd frame goes to `AppState` via the delegate.
2. `SegmentationEngine.segment()` runs VNGenerateForegroundInstanceMaskRequest and finds 3 objects in the frame (a dog, a couch, a lamp).
3. `InstanceTracker.update()` matches these to previously tracked instances via IoU. The dog has been stable for 2.1 seconds -- well past the 0.5s threshold. It's tappable.
4. Arya's finger lands on the screen. `ContentView` normalizes the tap coordinates (0-1 range) and calls `AppState.handleTap()`.
5. `InstanceTracker.instance(at:)` checks which tracked instance's bounding box contains the tap point. It's the dog.
6. A light haptic fires. Mode changes to `.classifying`.
7. `AppState.cropPixelBuffer()` extracts just the dog's bounding box region from the full-resolution frame.
8. `ClassificationEngine.classify()` fires both models in parallel:
   - VNClassify returns "golden retriever" at 0.91 confidence (2nd place: "Labrador" at 0.04)
   - MobileCLIP returns "dog" at 0.89 similarity (2nd place: "cat" at 0.31)
9. `LabelMapper` maps "golden retriever" to "dog".
10. `ConsensusGate` checks: both agree on "dog" -- accept. CLIP 0.89 >= 0.20 threshold -- pass.
11. Mode changes to `.learning(word: "dog", instanceIndex: 2)`.
12. `LearningOverlayView` appears. `MaskOverlayView` generates the dog's silhouette mask, dims everything else, adds gold glow.
13. `WordDisplayView` shows "d o g" in 80pt SF Rounded Bold, initially dim white.
14. `WordSpeaker.speak(word: "dog")` loads `Vocabulary/dog/audio.m4a` and starts playing.
15. `CADisplayLink` fires 60 times per second. Each tick, `WordSpeaker.currentTime` updates.
16. `LetterHighlighter.letterStates(at:)` maps the current time to phoneme boundaries from `timing.json` and returns `[.active, .upcoming, .upcoming]` → `[.spoken, .active, .upcoming]` → `[.spoken, .spoken, .active]` as each sound plays.
17. `WordDisplayView` animates each letter's color transition with 0.15s ease-in-out.
18. Audio finishes. All letters glow gold for 0.5 seconds.
19. Arya taps the screen. `AppState.dismissLearning()` stops audio and returns to `.exploring`.
20. The camera feed is pure again. Arya looks for the next thing to tap.

That's the whole app. One flow. Done right.

---

## Project Structure At a Glance

```
ARYA/
├── project.yml                  ← XcodeGen config (THE source of truth for project structure)
├── ARYA/
│   ├── App/
│   │   ├── ARYAApp.swift        ← Entry point, camera permission
│   │   └── AppState.swift       ← The conductor: owns all engines, manages state machine
│   ├── Data/
│   │   ├── VocabularyStore.swift ← Loads vocabulary.json
│   │   ├── LabelMapper.swift    ← VNClassify label → child word whitelist
│   │   ├── TimingData.swift     ← Codable model for phoneme timing
│   │   └── CLIPEmbeddings.swift ← Loads text embeddings, runs image encoder, cosine similarity
│   ├── Detection/
│   │   ├── CameraManager.swift        ← AVCaptureSession, frame delivery, frame skipping
│   │   ├── SegmentationEngine.swift   ← VNGenerateForegroundInstanceMaskRequest
│   │   ├── InstanceTracker.swift      ← IoU-based tracking, 0.5s stability gate
│   │   ├── ClassificationEngine.swift ← Dual-model (VNClassify + MobileCLIP) in parallel
│   │   └── ConsensusGate.swift        ← Confidence thresholds, margin checks, agreement
│   ├── Speech/
│   │   ├── WordSpeaker.swift       ← AVAudioPlayer + CADisplayLink
│   │   └── LetterHighlighter.swift ← Phoneme time → letter states
│   ├── Views/
│   │   ├── CameraPreviewView.swift    ← UIViewRepresentable for camera preview
│   │   ├── ContentView.swift          ← Root view: camera + overlays
│   │   ├── LearningOverlayView.swift  ← Dim + glow + word display
│   │   ├── MaskOverlayView.swift      ← Pixel-perfect object silhouette with CIFilter effects
│   │   ├── WordDisplayView.swift      ← Per-letter color animation
│   │   └── OnboardingHintView.swift   ← First-launch pulsing tap hint
│   └── Resources/
│       ├── Vocabulary/            ← 103 subdirectories, each with audio.m4a + timing.json
│       ├── vocabulary.json                    ← Master word list (107 words)
│       ├── label_mappings.json                ← VNClassify → child word whitelist
│       ├── text_embeddings.bin                ← Pre-computed MobileCLIP text embeddings (214KB)
│       └── MobileCLIPImageEncoder.mlpackage/  ← MobileCLIP S0 image encoder (22MB, compiled by Xcode)
├── ARYATests/                     ← Unit tests for logic components
└── scripts/                       ← Build-time audio generation pipeline
```

---

## Final Thought

The best children's apps feel inevitable -- like of *course* you'd tap a thing and hear its name. But behind that simplicity is a dual-model consensus pipeline, frame-perfect audio synchronization, pixel-level mask rendering, and a carefully curated whitelist of 107 words. The complexity exists so the child never has to experience it.

That's the job, really. Make the hard stuff invisible.
