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
 Data Layer        — What the app knows (vocabulary, corrections, timing data)
```

Why five layers instead of just writing everything in one ViewController like a cowboy? Because each layer has a fundamentally different job, and keeping them separate means you can change one without breaking the others. The detection layer doesn't know or care about SwiftUI. The speech layer has no idea there's a camera. AppState is the conductor of the orchestra -- it's the only thing that talks to everybody.

This is a pattern worth internalizing: **when you find yourself writing a class that does camera work AND plays audio AND manages UI state, you've gone wrong somewhere.** Split it up. Your future self will thank you.

---

## The UI Layer: Zero Chrome, Maximum Trust

Open `/ARYA/Views/ContentView.swift` and you'll notice something unusual: there's almost nothing there. No navigation bars. No buttons (almost). No menus. The camera IS the interface. This is a deliberate design philosophy called "zero chrome," and it's perfect for a 2-year-old who can't read a menu anyway.

The entire view hierarchy is a ZStack with these layers:

1. **CameraPreviewView** -- always visible, full screen, wrapping an AVCaptureVideoPreviewLayer
2. **TapGlowView** -- a radial glow effect at the tap point, shown during learning or when prompting for correction
3. **LearningOverlayView** -- appears when a word is being taught (shows the word with letter-by-letter highlighting)
4. **CorrectionPickerView** -- a scrollable word list that slides up from the bottom, letting a parent correct a wrong label or identify an unrecognized object
5. **OnboardingHintView** -- a pulsing tap icon shown exactly once on first launch, then gone forever

The `WordDisplayView` renders each letter individually in 80pt SF Rounded Bold with 16pt spacing between letters. Each letter has its own color state (dim white for upcoming, bright gold for currently speaking, softer gold for already spoken), and transitions happen with 0.15s ease-in-out animations. The effect is like a karaoke bouncing ball, but for learning the word "elephant."

### The Parent Correction Flow

When the app gets a word wrong, a parent can tap the pencil icon (bottom-right corner during learning) to bring up the `CorrectionPickerView`. This shows all 127 vocabulary words in a searchable, scrollable list. The parent picks the right word, and two things happen: (1) the app immediately starts teaching the correct word, and (2) it stores the 1024-dimensional feature embedding of that object alongside the correct label in `CorrectionStore`. Next time the child taps something that looks similar, the correction kicks in before the models even have a say.

This also works for objects the model doesn't recognize at all. When the consensus gate rejects a classification but the custom classifier still produced a feature embedding, the app shows the glow and the correction picker automatically -- inviting the parent to label the unknown object. The child sees a glow (something happened!), and the parent gets a chance to teach both the child and the model.

---

## The Detection Layer: Two Brains Are Better Than One

Here's where the engineering gets interesting. The detection layer has two subsystems that work completely differently, and they have to agree before anything is shown to a child.

### Subsystem A: Continuous Segmentation (The Eyes)

`SegmentationEngine.swift` runs Apple's `VNGenerateForegroundInstanceMaskRequest` on every 3rd camera frame. That's roughly 10 segmentations per second on a 30fps camera feed. It finds every distinct object in the frame and generates a pixel-perfect mask for each one.

But here's the thing -- the child never sees any of this. There are no outlines, no bounding boxes, no visual hints. The camera feed looks completely clean. The segmentation runs silently in the background like a guard dog watching but not barking.

`InstanceTracker.swift` then tracks these detected instances across frames using IoU (Intersection over Union) -- basically asking "is this the same object I saw last frame, just moved slightly?" An instance must be stable for 0.5 seconds before it becomes tappable. This prevents phantom objects from camera shake. Imagine a toddler's hand jiggling the phone -- without this stability requirement, the app would detect objects that flicker in and out of existence.

### Subsystem B: On-Tap Classification (The Brain)

Classification is the expensive part, so it only runs when the child actually taps something. This is a key architectural decision: **don't waste compute on things the user didn't ask about.** The child tapped ONE object. Classify that ONE object.

When a tap lands, `AppState` crops a region centered on the tap point (25% of the shorter buffer dimension) and sends it to `ClassificationEngine.classify()`, which runs TWO classifiers in parallel using Swift's `async let`:

1. **Apple's VNClassifyImageRequest** -- built into iOS, knows 1,303 categories. Very accurate but very specific. It'll say "golden retriever" when a kid just needs to hear "dog." Also has blind spots: screens, monitors, and similar objects often get unhelpful labels like "document" or "screenshot."

2. **ARYAClassifier** (Custom MobileNetV3-Small, 3.2MB CoreML model) -- trained on our own curated dataset of 127 child-vocabulary classes. Takes a 224x224 RGB image and outputs two things: a softmax probability distribution over all 127 classes, and a 1024-dimensional feature vector. The feature vector is the model's internal representation of the image -- think of it as a fingerprint that captures what the object "looks like" in a way that's useful for finding similar objects later.

We started with Apple's MobileCLIP S0 (a 22MB zero-shot CLIP model) as the second classifier. It worked, but had a fundamental limitation: it compared images against text descriptions ("a photo of a dog") rather than learning what each object actually looks like from training data. The custom MobileNetV3-Small replaced it because: (a) it's trained specifically on our 127 vocabulary words with curated images, (b) it's 7x smaller (3.2MB vs 22MB), (c) it produces a feature embedding we can use for the correction system, and (d) its softmax probabilities are easier to threshold than cosine similarities. The CLIP approach was a good stepping stone -- it let us ship something while we built the training pipeline -- but a purpose-trained model is always going to beat a general-purpose one on a specific task.

### The Consensus Gate: Trust, But Verify

`ConsensusGate.swift` is the bouncer at the door. It uses probability-based thresholds to decide when to trust which model:

- VNClassify's raw label (like "golden retriever") gets mapped to a child word (like "dog") through `LabelMapper`. Confidence is **aggregated** across all VN labels that map to the same word (so `cup` + `mug` both contribute to the "cup" score).
- **Custom very high confidence (>0.90)**: Accept the custom classifier's word outright. If the model trained on our data is 90%+ sure, trust it.
- **Custom medium confidence (0.40-0.90), models agree**: Accept. Both brains think it's the same thing.
- **Custom medium confidence (0.40-0.90), models disagree, VN strong (>0.45)**: Trust VN. Apple's model is a strong second opinion.
- **Custom medium confidence with strong margin, VN weak**: Trust custom. If custom says "dog" at 0.55 and the second choice is at 0.15 (margin 3.7x), and VN is unsure, the custom model is probably right.
- **Custom low confidence (<0.10)**: Fall back to VN alone, but only if VN is reasonably confident with margin.
- **Custom-only path** (VN has no mapped word): Accept custom if confidence > 0.45. This handles cases where VN returns useless labels like "document."
- **Neither model confident**: Reject silently. But if the custom classifier produced a feature embedding, pass it back so the parent can correct via CorrectionPickerView.

Before any of this runs, `CorrectionStore` gets first crack. If the image's feature embedding is >0.85 cosine similar to a stored correction, that correction's word wins immediately, bypassing both models. This is how parent corrections stick.

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

The playback rate is set to 0.85x on top of ElevenLabs' 0.7x generation speed, making pronunciation slow and clear enough for toddlers to hear each phoneme distinctly. `AVAudioPlayer.enableRate` must be set to `true` before setting the rate -- a detail that's easy to miss.

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

127 words, all concrete physical nouns a 2-4 year old encounters in daily life. Categories include animals (bear, cat, deer, dog, giraffe, panda, penguin, sheep, snake, tiger, zebra...), food (apple, banana, grape, kiwi, mango, pineapple, strawberry, watermelon...), home items (bathtub, bed, chair, lamp, stairs, toilet paper...), kitchen items (bowl, cup, fork, plate...), outdoor things (ball, bicycle, car, flower, tree...), clothing (hat, shirt, shoe...), and more. Every word was chosen to be: pronounceable by a toddler, generic (never "Tesla" -- always "car"), and physically present in typical daily life.

### CorrectionStore: Learning from Mistakes

`CorrectionStore.swift` is one of the cleverest parts of the system. When a parent corrects a misidentification, the store saves the 1024-dimensional feature embedding (from the custom classifier's penultimate layer) alongside the correct word. These corrections persist to disk as JSON in the app's documents directory.

On the next tap, before the consensus gate even runs, the store compares the new image's embedding against all stored corrections using cosine similarity. If any correction matches above 0.85 threshold, that word wins immediately. The embedding comparison uses the `cosineSimilarity` function from `ImageUtils.swift`, which computes the dot product of two L2-normalized vectors.

This is a form of few-shot learning -- the parent provides one example ("this is a panda"), and the 1024-dim feature space is rich enough that similar-looking objects will match in the future. The 0.85 threshold is high enough to avoid false matches but low enough that the same object from a slightly different angle still hits.

One important gotcha: if you retrain the custom classifier and the embedding dimensions change (or the feature space shifts significantly), existing corrections become invalid. The user needs to clear their corrections after a model update.

### Build-Time Audio Pipeline

This is the part that lives outside the app, in the `scripts/` directory. The pipeline goes:

1. **ElevenLabs API** generates voice recordings for each word at 0.7x speed (warm, friendly, slow enough for toddlers). Voice: Jessica (`cgSgspJ2msm6clMCkdW9`) -- "Playful, Bright, Warm."
2. **`build_timing_from_audio.py`** generates timing data by proportionally mapping phoneme durations to the actual audio length, using weighted phoneme profiles (vowels get more time than stops).
3. Hand-verified phoneme-to-letter mappings (from CMU Pronouncing Dictionary) in `phoneme_to_letter_mappings.json` ensure things like "ph" -> one sound are correct.

All 127 words have pre-generated audio and timing data bundled into the app. The app is fully offline -- no internet needed, ever.

Note: Montreal Forced Aligner (MFA) can produce sub-20ms phoneme boundaries from real audio analysis, but it requires a conda environment and isn't currently installed. The proportional timing approach works well enough for the current vocabulary. If you need higher accuracy, install MFA and run `scripts/align_audio.py` followed by `scripts/build_timing_data.py`.

### The Training Pipeline

The custom classifier doesn't train itself. There's a four-stage pipeline in `scripts/`:

1. **`collect_training_data.py`** -- Downloads training images from Open Images V7 and COCO via the `fiftyone` library. Each vocabulary word maps to one or more dataset classes (e.g., "deer" pulls from Open Images' "Deer" class; "bag" pulls from "Handbag"). Images are cropped to object bounding boxes and saved as JPGs. Some words (cloud, star, rock, berry varieties) need manual image collection because they aren't well-represented in standard datasets.

2. **`curate_training_data.py`** -- Quality control. Removes duplicates, filters out images that are too small or too blurry, and ensures each class has enough samples. This step matters more than you'd think -- noisy training data is the #1 cause of confused classifiers.

3. **`train_classifier.py`** -- Fine-tunes MobileNetV3-Small (pretrained on ImageNet) using PyTorch. The final layer is replaced with a 127-class head. Training produces both softmax probabilities and a 1024-dim feature vector from the penultimate layer. Class names use spaces (not underscores) to match `vocabulary.json` -- PyTorch's `ImageFolder` uses directory names as classes, and `train_classifier.py` explicitly replaces underscores with spaces during loading.

4. **`convert_to_coreml.py`** -- Converts the PyTorch model to CoreML `.mlpackage` format using `coremltools`. The model is quantized to INT8 for smaller size and faster Neural Engine inference. The output is `ARYAClassifier.mlpackage`, which Xcode compiles to `.mlmodelc` at build time.

There's also `verify_classifier.py` for spot-checking the converted model against known test images.

---

## The Orchestration: AppState

`AppState.swift` is the conductor. It's a `@MainActor` `ObservableObject` that owns every major component:

- `CameraManager` -- delivers frames
- `SegmentationEngine` -- finds objects in frames
- `InstanceTracker` -- stabilizes detections
- `ClassificationEngine` -- identifies tapped objects (which owns `CustomClassifier`, `LabelMapper`, `ConsensusGate`, and `CorrectionStore`)
- `WordSpeaker` -- plays audio
- `VocabularyStore` -- knows all the words

The state machine has three modes: `exploring` -> `classifying` -> `learning(word, instanceIndex)` -> back to `exploring`. If classification fails at any point, it silently returns to `exploring`. No error dialogs. No "sorry, we couldn't identify that." The only exception: if classification fails but produced a feature embedding, the app shows the correction picker so a parent can label the object.

The `handleTap` method is worth studying. It checks the mode, fires a haptic, sets mode to `.classifying`, crops the pixel buffer around the tap point, runs dual classification through async/await, and then branches:
- If a word was identified: transition to `.learning(word:instanceIndex:)`.
- If no word but features exist: show `CorrectionPickerView` for parent labeling.
- If nothing useful came back: silently return to `.exploring`.

A `showGlow` computed property controls when the `TapGlowView` appears -- during learning mode OR when the correction picker is showing. This avoids duplicating the glow logic across multiple conditions.

---

## Technologies Used (And Why)

| Technology | Why This One |
|---|---|
| **SwiftUI** | The overlay animations (per-letter color transitions, opacity fades) are trivially declarative. UIKit would require significantly more animation management code. |
| **Vision Framework** | `VNGenerateForegroundInstanceMaskRequest` gives pixel-perfect object silhouettes for free (no ML model needed). `VNClassifyImageRequest` provides 1,303-category classification built into iOS. No downloads, no API keys. |
| **CoreML + MobileNetV3-Small** | Custom-trained 3.2MB classifier with 127 classes. Outputs softmax probabilities (for thresholding) and 1024-dim feature vectors (for CorrectionStore). Runs on Neural Engine. INT8 quantized for speed and size. |
| **AVFoundation** | The only real option for camera access on iOS. We need raw pixel buffers (not just a viewfinder) so we can run Vision requests on them. |
| **AVAudioPlayer + CADisplayLink** | `AVAudioPlayer.currentTime` gives ground-truth playback position. `CADisplayLink` fires at display refresh rate (~60fps). Together they provide frame-accurate audio-visual sync without timer drift. `enableRate` allows playback speed control for clearer pronunciation. |
| **XcodeGen** | The `project.yml` file is 40 lines. The generated `.xcodeproj` is thousands of lines of XML. Version-controlling a YAML file instead of an Xcode project file prevents merge conflicts and makes the project definition human-readable. |
| **PyTorch + coremltools** | Training pipeline: fine-tune MobileNetV3-Small on curated data, convert to CoreML with INT8 quantization. The PyTorch ecosystem has the best training tooling; CoreML has the best on-device inference on Apple hardware. Use each where it's strongest. |
| **fiftyone** | Open-source dataset library that provides easy access to Open Images V7 and COCO. One API call to download images with bounding boxes, filter by class, and crop to objects. Saved weeks of manual data collection. |
| **ElevenLabs** | Natural-sounding TTS with speed control. Voice "Jessica" at 0.7x speed produces warm, clear pronunciation. The free tier (10,000 chars/month) covers the whole vocabulary. |

---

## Bugs We Hit and How We Fixed Them

### 1. The Model Confidence Trap

Early in device testing, the custom classifier reported "cat" at 0.923 confidence for what was clearly a panda stuffed animal. The gut reaction was "well, the model is very confident, so it must be right." Wrong. The panda class didn't exist yet in the 107-word vocabulary -- the model had no choice but to pick the closest thing it knew, and it picked "cat" with high confidence because that was the best match in its limited world.

This is the single most important lesson from the project: **high model confidence does not mean correctness.** A model can only choose from the classes it was trained on. If the right answer isn't in the class list, the model will confidently pick the wrong one. Always validate against ground truth. The fix was adding "panda" (and 19 other classes) to the vocabulary and retraining.

### 2. The MobileCLIP to Custom Classifier Migration

We initially used Apple's MobileCLIP S0 for zero-shot classification. It worked by comparing image embeddings against pre-computed text embeddings ("a photo of a dog") using cosine similarity. The approach had three problems: (a) the 22MB model was large, (b) cosine similarity thresholds were hard to tune because the similarity values are less interpretable than softmax probabilities, and (c) it confused similar-looking things that have different names (monitors vs TVs, cups vs mugs) because it was matching against text descriptions rather than learning visual features from labeled examples.

The custom MobileNetV3-Small solved all three: 3.2MB, clean softmax probabilities, and trained on curated images of the actual objects kids encounter. The migration required removing three dead files (`CLIPEmbeddings.swift`, `MobileCLIPImageEncoder.mlpackage`, `text_embeddings.bin`) and rewiring `ClassificationEngine` to use `CustomClassifier` instead.

### 3. The WordSpeaker NSObject Saga

`WordSpeaker` needs to be an `AVAudioPlayerDelegate` to know when audio finishes playing. `AVAudioPlayerDelegate` is an Objective-C protocol, which means the conforming class must inherit from `NSObject`. But we also needed it to be an `ObservableObject` to publish `currentTime` to SwiftUI.

The fix: `final class WordSpeaker: NSObject, ObservableObject`. This works because `NSObject` is a class and `ObservableObject` is a protocol. Swift allows a class to inherit from one class and conform to multiple protocols. But you have to put the class first in the declaration -- `NSObject, ObservableObject` -- not the other way around.

**Lesson:** When bridging Swift and Objective-C patterns (which happens a lot with AVFoundation and UIKit), you'll frequently need NSObject as a base class. Know when and why.

### 4. The Vocabulary Folder Reference Collision

XcodeGen was trying to compile the `Vocabulary/` folder contents as Swift source files. Audio files and JSON files being fed to the Swift compiler. Naturally, it exploded.

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

**Lesson:** XcodeGen's `type: folder` creates a folder reference (blue folder in Xcode) instead of a group (yellow folder). Folder references include ALL contents at build time without listing each file individually. This is essential when you have 127 subdirectories each containing audio and JSON files.

### 5. The Glow Position Offset Bug

The `TapGlowView` appeared 59 points above where the user actually tapped. The cause: `ContentView` uses `.ignoresSafeArea()`, but the glow view's coordinate system still accounted for the 59pt status bar. When two views overlay each other and one ignores safe area while the other doesn't, their coordinate origins diverge silently.

**Lesson:** When multiple views overlay each other in a ZStack, confirm they share the same coordinate space. Safe area insets are the #1 cause of "everything is offset by a mysterious fixed amount" bugs on iOS.

### 6. MLMultiArray Float16 on ANE

The custom classifier runs on the Apple Neural Engine, which outputs MLMultiArray values in Float16 format. The naive approach -- `dataPointer.bindMemory(to: Float.self)` -- reads garbage because it interprets 2-byte Float16 as 4-byte Float32. The correct approach: use subscript access (`array[i].floatValue`), which handles the type conversion automatically.

**Lesson:** Never assume the numeric type of MLMultiArray data. Check `dataType` at runtime and use subscript access unless you've explicitly verified the type.

### 7. CameraManager Delegate Threading

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

### 8. SwiftUI Pattern Matching in View Builders

Tried to write `if case .learning = mode || showingCorrectionPicker` to show the glow in two different states. Swift's pattern matching syntax can't be combined with `||` inside SwiftUI view builders. The compiler error is unhelpful.

The fix: extract the condition into a computed property (`showGlow`) on AppState. Clean, readable, and avoids fighting the view builder DSL.

---

## Best Practices and How Good Engineers Think

### 1. "No Error States" Philosophy

This app has zero error dialogs. Zero loading spinners. Zero "something went wrong" messages. If classification fails, nothing happens (or the correction picker appears for a parent to help). If the camera can't start, the screen is just black. If audio files are missing, the word appears without sound.

This isn't laziness -- it's intentional design for a 2-year-old user. A toddler doesn't understand "classification confidence below threshold." They just tap something else. Every possible failure mode silently degrades to a reasonable state. This is called **graceful degradation** and it's especially important in apps for users who can't read error messages.

### 2. Conservative Thresholds for Children

The ConsensusGate thresholds (customHighConfidence=0.90, customMediumConfidence=0.40, customLowConfidence=0.10, customOnlyConfidence=0.45, vnTrustThreshold=0.45) are calibrated to reject ambiguous results while still recognizing most common objects. We'd rather miss some valid objects than mislabel a single one. In an adult app, you might show a "did you mean...?" prompt. A 2-year-old can't evaluate whether a suggestion is correct. Whatever the app says, they'll believe it.

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

### 7. Train on Your Actual Domain

The biggest accuracy improvement didn't come from tuning thresholds or swapping model architectures. It came from training a classifier specifically on images of the objects kids encounter (household items, common animals, everyday food) rather than relying on a general-purpose model that knows 1,303 categories or a CLIP model matching text descriptions. A 3.2MB purpose-trained model outperforms a 22MB general-purpose one for this specific task. Match your model to your problem.

---

## Potential Pitfalls (And How to Avoid Them)

### Simulator Won't Work

`VNGenerateForegroundInstanceMaskRequest` requires a physical device running iOS 17+. If you try to run on the simulator, segmentation silently returns empty results. There's no crash, no error -- just no objects detected. Always test on a real device.

### Class Naming Must Match Everywhere

The vocabulary word list (`vocabulary.json`), the class list (`arya_classes.json`), the training data directories, and the label mappings (`label_mappings.json`) all must use the same names. A mismatch anywhere in the chain causes silent failures: the model outputs a class name that doesn't match the vocabulary, so the word is never shown. PyTorch's `ImageFolder` uses directory names as class labels; `train_classifier.py` replaces underscores with spaces to match. If you add a new word, update all four files.

### Embedding Dimension Changes Invalidate Corrections

`CorrectionStore` saves 1024-dim feature vectors from the custom classifier. If you retrain the model and the feature dimensions change (or even if the feature space shifts significantly due to different training data), existing corrections become meaningless -- the cosine similarity comparisons will produce garbage. Warn users to clear their corrections after a model update.

### Memory Pressure from CVPixelBuffers

`CVPixelBuffer` objects from the camera can be large (1920x1080 BGRA = ~8MB each). The frame-skipping and `alwaysDiscardsLateVideoFrames = true` help, but be careful about retaining pixel buffers longer than necessary. The segmentation result stores a reference to the pixel buffer for mask generation, so it stays alive as long as `liveSegmentation` is set.

### Audio Session Configuration

`WordSpeaker` sets the audio session category to `.playback` with `.mixWithOthers` before each play. `.playback` means audio plays even when the phone is in silent mode -- which is correct, since teaching a word IS the primary purpose. `.mixWithOthers` prevents the audio from interrupting the camera capture session. Using `.ambient` instead would cause audio to be silenced by the ringer switch, which is wrong for this app.

### CIContext Reuse

`ImageUtils.swift` provides a shared `CIContext` (`sharedCIContext`) for all Core Image work. Creating a new `CIContext` for every operation is extremely expensive (GPU resource allocation, shader pipeline caching). If you add new image processing code, use the shared context.

### Timing Data Must Match Audio

If you regenerate audio files without regenerating timing data (or vice versa), the letter highlights will be out of sync. The `scripts/generate_audio.py` and `scripts/build_timing_from_audio.py` should be run together for any changed words.

### The "Phase Problem" in Phoneme Mapping

Some English words have ambiguous phoneme-to-letter mappings. Consider "knight" -- is the "n" sound mapped to the letter "k" (which is silent) or "n"? The CMU Pronouncing Dictionary handles pronunciation, but the *mapping back to letters* requires manual verification. All 127 words have been hand-verified in `phoneme_to_letter_mappings.json`, but if you add new words, expect to spend a few minutes per word checking these mappings.

---

## How the Pieces Connect (The Full Flow)

Let's trace what happens when a 3-year-old named Arya taps a dog on screen:

1. `CameraManager` is delivering 30fps frames. Every 3rd frame goes to `AppState` via the delegate.
2. `SegmentationEngine.segment()` runs VNGenerateForegroundInstanceMaskRequest and finds 3 objects in the frame (a dog, a couch, a lamp).
3. `InstanceTracker.update()` matches these to previously tracked instances via IoU. The dog has been stable for 2.1 seconds -- well past the 0.5s threshold. It's tappable.
4. Arya's finger lands on the screen. `ContentView` converts the screen point to image coordinates via `CameraManager.imagePoint(fromScreenPoint:)` and calls `AppState.handleTap()`.
5. A light haptic fires. Mode changes to `.classifying`.
6. `AppState` computes a tap-centered crop rect (25% of shorter dimension) and crops the pixel buffer.
7. `ClassificationEngine.classify()` first checks `CorrectionStore` -- no match. Then fires both models in parallel:
   - VNClassify returns "golden retriever" at 0.91 confidence
   - Custom classifier returns "dog" at 0.87 probability (second: "cat" at 0.03), plus a 1024-dim feature vector
8. `LabelMapper` maps "golden retriever" to "dog".
9. `ConsensusGate.evaluate()` checks: custom at 0.87 (medium range), VN agrees on "dog" -- accept.
10. `ClassificationResult(word: "dog", features: [...])` is returned. `lastFeatures` is saved for potential correction.
11. Mode changes to `.learning(word: "dog", instanceIndex: 0)`.
12. `TapGlowView` appears at the tap point.
13. `LearningOverlayView` shows "d o g" in 80pt SF Rounded Bold, initially dim white.
14. `WordSpeaker.speak(word: "dog")` loads `Vocabulary/dog/audio.m4a`, sets rate to 0.85x, and starts playing.
15. `CADisplayLink` fires 60 times per second. Each tick, `WordSpeaker.currentTime` updates.
16. `LetterHighlighter.letterStates(at:)` maps the current time to phoneme boundaries from `timing.json` and returns `[.active, .upcoming, .upcoming]` -> `[.spoken, .active, .upcoming]` -> `[.spoken, .spoken, .active]` as each sound plays.
17. `WordDisplayView` animates each letter's color transition with 0.15s ease-in-out.
18. Audio finishes. All letters glow gold for 0.5 seconds.
19. Arya taps the screen. `AppState.dismissLearning()` stops audio and returns to `.exploring`.
20. The camera feed is pure again. Arya looks for the next thing to tap.

That's the whole app. One flow. Done right.

---

## Project Structure At a Glance

```
ARYA/
├── project.yml                  <- XcodeGen config (THE source of truth for project structure)
├── ARYA/
│   ├── App/
│   │   ├── ARYAApp.swift        <- Entry point, camera permission
│   │   └── AppState.swift       <- The conductor: owns all engines, manages state machine
│   ├── Data/
│   │   ├── VocabularyStore.swift <- Loads vocabulary.json
│   │   ├── LabelMapper.swift    <- VNClassify label -> child word whitelist
│   │   ├── TimingData.swift     <- Codable model for phoneme timing
│   │   ├── CorrectionStore.swift <- Stores parent corrections (embedding + word pairs)
│   │   └── ImageUtils.swift     <- Shared CIContext, cosineSimilarity, pixel buffer helpers
│   ├── Detection/
│   │   ├── CameraManager.swift        <- AVCaptureSession, frame delivery, frame skipping
│   │   ├── SegmentationEngine.swift   <- VNGenerateForegroundInstanceMaskRequest
│   │   ├── InstanceTracker.swift      <- IoU-based tracking, 0.5s stability gate
│   │   ├── ClassificationEngine.swift <- Dual-model (VNClassify + custom) with CorrectionStore
│   │   ├── CustomClassifier.swift     <- MobileNetV3-Small CoreML wrapper (3.2MB, 127 classes)
│   │   └── ConsensusGate.swift        <- Probability thresholds, margin checks, agreement logic
│   ├── Speech/
│   │   ├── WordSpeaker.swift       <- AVAudioPlayer + CADisplayLink + rate control
│   │   └── LetterHighlighter.swift <- Phoneme time -> letter states
│   ├── Views/
│   │   ├── CameraPreviewView.swift    <- UIViewRepresentable for camera preview
│   │   ├── ContentView.swift          <- Root view: camera + overlays + correction flow
│   │   ├── LearningOverlayView.swift  <- Word display during learning
│   │   ├── TapGlowView.swift         <- Radial glow at tap point
│   │   ├── WordDisplayView.swift      <- Per-letter color animation
│   │   ├── CorrectionPickerView.swift <- Searchable word list for parent corrections
│   │   └── OnboardingHintView.swift   <- First-launch pulsing tap hint
│   └── Resources/
│       ├── Vocabulary/                         <- 127 subdirectories, each with audio.m4a + timing.json
│       ├── vocabulary.json                     <- Master word list (127 words)
│       ├── label_mappings.json                 <- VNClassify -> child word whitelist
│       ├── arya_classes.json                   <- Ordered class list matching model output indices
│       └── ARYAClassifier.mlpackage/           <- Custom MobileNetV3-Small (3.2MB, compiled by Xcode)
├── ARYATests/                     <- Unit tests (151 tests covering logic components)
└── scripts/                       <- Training pipeline + audio generation
    ├── collect_training_data.py   <- Downloads from Open Images V7 + COCO via fiftyone
    ├── curate_training_data.py    <- Deduplication, quality filtering
    ├── train_classifier.py        <- Fine-tune MobileNetV3-Small, export PyTorch model
    ├── convert_to_coreml.py       <- PyTorch -> CoreML with INT8 quantization
    ├── verify_classifier.py       <- Spot-check converted model
    ├── generate_audio.py          <- ElevenLabs TTS for vocabulary words
    └── build_timing_from_audio.py <- Generate phoneme timing from audio duration
```

---

## Final Thought

The best children's apps feel inevitable -- like of *course* you'd tap a thing and hear its name. But behind that simplicity is a dual-model consensus pipeline, a parent correction system that learns from one example, frame-perfect audio synchronization, and a carefully curated whitelist of 127 words. The complexity exists so the child never has to experience it.

That's the job, really. Make the hard stuff invisible.
