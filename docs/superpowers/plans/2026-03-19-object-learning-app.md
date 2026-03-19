# ARYA — Object Learning App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an iOS camera app for children (ages 2-4) that identifies real-world objects via tap, highlights them with pixel-perfect silhouettes, and teaches the word through synchronized voice pronunciation with letter-level highlighting.

**Architecture:** Five-layer architecture (UI → Interaction → Detection → Speech → Data). Camera segmentation runs continuously but classification only fires on-tap for maximum accuracy. Dual-model consensus (VNClassify + MobileCLIP) ensures correct labeling. Pre-recorded audio with forced-alignment timing data drives letter-level pronunciation sync.

**Tech Stack:** Swift/SwiftUI, AVFoundation (camera), Vision framework (segmentation + classification), CoreML (MobileCLIP), AVAudioPlayer + CADisplayLink (speech sync), Python (build-time audio pipeline), ElevenLabs (TTS), Montreal Forced Aligner (phoneme timing)

**Spec:** `docs/superpowers/specs/2026-03-19-object-learning-app-design.md`

---

## File Structure

```
ARYA/
├── project.yml                              (xcodegen project definition)
├── ARYA/
│   ├── App/
│   │   ├── ARYAApp.swift                    (app entry, camera permission, single-view)
│   │   └── AppState.swift                   (ObservableObject: exploring/classifying/learning states)
│   ├── Data/
│   │   ├── VocabularyStore.swift            (loads vocabulary.json, provides word list)
│   │   ├── LabelMapper.swift                (VNClassify label → child word, whitelist logic)
│   │   ├── TimingData.swift                 (Codable model for phoneme timing JSON)
│   │   └── CLIPEmbeddings.swift             (loads pre-computed text embeddings, cosine similarity)
│   ├── Detection/
│   │   ├── CameraManager.swift              (AVCaptureSession, frame delivery via delegate)
│   │   ├── SegmentationEngine.swift         (VNGenerateForegroundInstanceMaskRequest, mask extraction)
│   │   ├── InstanceTracker.swift            (tracks instances across frames via IoU, enforces 0.5s stability)
│   │   ├── ClassificationEngine.swift       (runs VNClassify + MobileCLIP in parallel on cropped image)
│   │   └── ConsensusGate.swift              (confidence thresholds, margin checks, agreement verification)
│   ├── Speech/
│   │   ├── WordSpeaker.swift                (AVAudioPlayer + CADisplayLink, reports current phoneme index)
│   │   └── LetterHighlighter.swift          (maps phoneme index → letter indices, manages dim/active/spoken states)
│   ├── Views/
│   │   ├── CameraPreviewView.swift          (UIViewRepresentable wrapping AVCaptureVideoPreviewLayer)
│   │   ├── ContentView.swift                (root view: camera + overlays based on AppState)
│   │   ├── LearningOverlayView.swift        (dim mask + glow + word display, tap to dismiss)
│   │   ├── WordDisplayView.swift            (renders letters with per-letter color/glow animation)
│   │   └── OnboardingHintView.swift         (pulsing tap icon, shown once, stored in UserDefaults)
│   ├── Resources/
│   │   ├── Vocabulary/                      (per-word subdirectories with audio.m4a + timing.json)
│   │   ├── vocabulary.json                  (master word list with categories)
│   │   ├── label_mappings.json              (VNClassify → child word whitelist)
│   │   └── text_embeddings.bin              (pre-computed MobileCLIP text embeddings)
│   └── Info.plist
├── ARYATests/
│   ├── LabelMapperTests.swift
│   ├── TimingDataTests.swift
│   ├── ConsensusGateTests.swift
│   ├── InstanceTrackerTests.swift
│   └── LetterHighlighterTests.swift
├── scripts/
│   ├── generate_audio.py                    (ElevenLabs API → audio files)
│   ├── align_audio.py                       (Montreal Forced Aligner → TextGrid)
│   ├── build_timing_data.py                 (TextGrid → timing.json per word)
│   ├── build_label_mappings.py              (generate label_mappings.json from VNClassify taxonomy)
│   ├── build_clip_embeddings.py             (generate text_embeddings.bin via MobileCLIP text encoder)
│   ├── phoneme_to_letter_mappings.json      (hand-verified phoneme→letter index maps per word)
│   └── requirements.txt
└── docs/
```

**Note on testing:** Data layer and logic components (LabelMapper, TimingData, ConsensusGate, InstanceTracker, LetterHighlighter) are fully unit-testable. Camera, Vision, and CoreML components require a physical device and will be tested manually during integration (Task 14). TDD applies where tests are meaningful.

---

## Task 1: Project Setup

**Files:**
- Create: `project.yml`, `ARYA/App/ARYAApp.swift`, `ARYA/Info.plist`, `.gitignore`

- [ ] **Step 1: Install xcodegen if needed**

```bash
which xcodegen || brew install xcodegen
```

- [ ] **Step 2: Create .gitignore**

```gitignore
# Xcode
*.xcodeproj/xcuserdata/
*.xcworkspace/xcuserdata/
DerivedData/
build/
*.hmap
*.ipa
*.dSYM.zip
*.dSYM

# Dependencies
Pods/
Carthage/

# macOS
.DS_Store

# Python
__pycache__/
*.pyc
venv/

# Project
.superpowers/
```

- [ ] **Step 3: Create project.yml**

```yaml
name: ARYA
options:
  bundleIdPrefix: com.arya
  deploymentTarget:
    iOS: "17.0"
  xcodeVersion: "16.0"
settings:
  base:
    SWIFT_VERSION: "5.9"
    TARGETED_DEVICE_FAMILY: 1
targets:
  ARYA:
    type: application
    platform: iOS
    sources:
      - path: ARYA
    settings:
      base:
        INFOPLIST_FILE: ARYA/Info.plist
        PRODUCT_BUNDLE_IDENTIFIER: com.arya.objectlearning
        ASSETCATALOG_COMPILER_APPICON_NAME: AppIcon
    scheme:
      testTargets:
        - ARYATests
  ARYATests:
    type: bundle.unit-test
    platform: iOS
    sources:
      - path: ARYATests
    dependencies:
      - target: ARYA
    settings:
      base:
        INFOPLIST_FILE: ARYATests/Info.plist
```

- [ ] **Step 4: Create Info.plist with camera permission**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSCameraUsageDescription</key>
    <string>ARYA uses the camera to help your child learn words by looking at objects around them.</string>
    <key>UILaunchScreen</key>
    <dict>
        <key>UIColorName</key>
        <string>LaunchScreenBackground</string>
    </dict>
    <key>UISupportedInterfaceOrientations</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
    </array>
    <key>UIRequiresFullScreen</key>
    <true/>
</dict>
</plist>
```

- [ ] **Step 5: Create minimal ARYAApp.swift**

```swift
import SwiftUI

@main
struct ARYAApp: App {
    var body: some Scene {
        WindowGroup {
            Text("ARYA")
                .font(.largeTitle)
        }
    }
}
```

- [ ] **Step 6: Create test Info.plist and placeholder test**

Create `ARYATests/Info.plist` (minimal plist) and `ARYATests/PlaceholderTests.swift`:

```swift
import XCTest
@testable import ARYA

final class PlaceholderTests: XCTestCase {
    func testAppLaunches() {
        XCTAssertTrue(true)
    }
}
```

- [ ] **Step 7: Generate Xcode project and verify build**

```bash
cd ARYA && xcodegen generate
xcodebuild -project ARYA.xcodeproj -scheme ARYA -destination 'platform=iOS Simulator,name=iPhone 16' build
```

Expected: BUILD SUCCEEDED

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: initialize ARYA Xcode project with xcodegen"
```

---

## Task 2: Data Layer — Vocabulary & Label Mappings

**Files:**
- Create: `ARYA/Data/VocabularyStore.swift`, `ARYA/Data/LabelMapper.swift`
- Create: `ARYA/Resources/vocabulary.json`, `ARYA/Resources/label_mappings.json`
- Test: `ARYATests/LabelMapperTests.swift`

- [ ] **Step 1: Create vocabulary.json (starter set — 5 words for development)**

```json
[
  { "word": "dog", "category": "animals" },
  { "word": "car", "category": "outdoors" },
  { "word": "cup", "category": "kitchen" },
  { "word": "chair", "category": "home" },
  { "word": "apple", "category": "food" }
]
```

- [ ] **Step 2: Create label_mappings.json (starter set)**

```json
{
  "golden retriever": "dog",
  "labrador retriever": "dog",
  "german shepherd": "dog",
  "poodle": "dog",
  "beagle": "dog",
  "puppy": "dog",
  "canine": "dog",
  "sedan": "car",
  "sports car": "car",
  "convertible": "car",
  "minivan": "car",
  "SUV": "car",
  "pickup truck": "car",
  "automobile": "car",
  "coffee cup": "cup",
  "teacup": "cup",
  "mug": "cup",
  "coffee mug": "cup",
  "drinking cup": "cup",
  "folding chair": "chair",
  "rocking chair": "chair",
  "office chair": "chair",
  "armchair": "chair",
  "dining chair": "chair",
  "Granny Smith": "apple",
  "red apple": "apple",
  "green apple": "apple",
  "eating apple": "apple"
}
```

- [ ] **Step 3: Write failing tests for LabelMapper**

```swift
// ARYATests/LabelMapperTests.swift
import XCTest
@testable import ARYA

final class LabelMapperTests: XCTestCase {

    var mapper: LabelMapper!

    override func setUp() {
        super.setUp()
        mapper = LabelMapper.load(from: Bundle.main)
    }

    func testMapsDetailedLabelToChildWord() {
        XCTAssertEqual(mapper.childWord(for: "golden retriever"), "dog")
        XCTAssertEqual(mapper.childWord(for: "sedan"), "car")
        XCTAssertEqual(mapper.childWord(for: "coffee mug"), "cup")
    }

    func testReturnsNilForUnmappedLabel() {
        XCTAssertNil(mapper.childWord(for: "espresso machine"))
        XCTAssertNil(mapper.childWord(for: "unknown thing"))
    }

    func testReturnsNilForExplicitlyRejectedLabel() {
        // If label_mappings.json contains null values, they should be rejected
        XCTAssertNil(mapper.childWord(for: "nonexistent"))
    }
}
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
xcodebuild test -project ARYA.xcodeproj -scheme ARYA -destination 'platform=iOS Simulator,name=iPhone 16'
```

Expected: FAIL — `LabelMapper` type not found

- [ ] **Step 5: Implement VocabularyStore**

```swift
// ARYA/Data/VocabularyStore.swift
import Foundation

struct VocabularyEntry: Codable {
    let word: String
    let category: String
}

final class VocabularyStore {
    let entries: [VocabularyEntry]
    let words: Set<String>

    init(entries: [VocabularyEntry]) {
        self.entries = entries
        self.words = Set(entries.map(\.word))
    }

    static func load(from bundle: Bundle = .main) -> VocabularyStore {
        guard let url = bundle.url(forResource: "vocabulary", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let entries = try? JSONDecoder().decode([VocabularyEntry].self, from: data) else {
            fatalError("Failed to load vocabulary.json")
        }
        return VocabularyStore(entries: entries)
    }

    func contains(_ word: String) -> Bool {
        words.contains(word)
    }
}
```

- [ ] **Step 6: Implement LabelMapper**

```swift
// ARYA/Data/LabelMapper.swift
import Foundation

final class LabelMapper {
    private let mappings: [String: String?]

    init(mappings: [String: String?]) {
        self.mappings = mappings
    }

    static func load(from bundle: Bundle = .main) -> LabelMapper {
        guard let url = bundle.url(forResource: "label_mappings", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            fatalError("Failed to load label_mappings.json")
        }
        var mappings: [String: String?] = [:]
        for (key, value) in raw {
            if let str = value as? String {
                mappings[key] = str
            } else {
                mappings[key] = nil as String?
            }
        }
        return LabelMapper(mappings: mappings)
    }

    /// Returns the child-friendly word for a VNClassify label, or nil if unmapped/rejected.
    func childWord(for classifierLabel: String) -> String? {
        guard let mapping = mappings[classifierLabel] else {
            return nil // Not in whitelist
        }
        return mapping // nil if explicitly rejected, String if mapped
    }
}
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
xcodebuild test -project ARYA.xcodeproj -scheme ARYA -destination 'platform=iOS Simulator,name=iPhone 16'
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: add VocabularyStore and LabelMapper with tests"
```

---

## Task 3: Data Layer — Timing Data Model

**Files:**
- Create: `ARYA/Data/TimingData.swift`
- Create: `ARYA/Resources/Vocabulary/dog/timing.json` (test fixture)
- Test: `ARYATests/TimingDataTests.swift`

- [ ] **Step 1: Create test timing.json for "dog"**

Place in `ARYA/Resources/Vocabulary/dog/timing.json`:

```json
{
  "word": "dog",
  "phonemes": [
    { "phoneme": "D",  "letters": [0], "start": 0.00, "end": 0.40 },
    { "phoneme": "AO", "letters": [1], "start": 0.40, "end": 0.95 },
    { "phoneme": "G",  "letters": [2], "start": 0.95, "end": 1.40 }
  ]
}
```

- [ ] **Step 2: Write failing tests for TimingData**

```swift
// ARYATests/TimingDataTests.swift
import XCTest
@testable import ARYA

final class TimingDataTests: XCTestCase {

    func testDecodesTimingJSON() throws {
        let json = """
        {
          "word": "dog",
          "phonemes": [
            { "phoneme": "D",  "letters": [0], "start": 0.00, "end": 0.40 },
            { "phoneme": "AO", "letters": [1], "start": 0.40, "end": 0.95 },
            { "phoneme": "G",  "letters": [2], "start": 0.95, "end": 1.40 }
          ]
        }
        """.data(using: .utf8)!

        let timing = try JSONDecoder().decode(TimingData.self, from: json)
        XCTAssertEqual(timing.word, "dog")
        XCTAssertEqual(timing.phonemes.count, 3)
        XCTAssertEqual(timing.phonemes[0].phoneme, "D")
        XCTAssertEqual(timing.phonemes[0].letters, [0])
        XCTAssertEqual(timing.phonemes[0].start, 0.0, accuracy: 0.001)
        XCTAssertEqual(timing.phonemes[0].end, 0.40, accuracy: 0.001)
    }

    func testMultiLetterPhoneme() throws {
        let json = """
        {
          "word": "elephant",
          "phonemes": [
            { "phoneme": "F", "letters": [3, 4], "start": 0.78, "end": 1.10 }
          ]
        }
        """.data(using: .utf8)!

        let timing = try JSONDecoder().decode(TimingData.self, from: json)
        XCTAssertEqual(timing.phonemes[0].letters, [3, 4])
    }

    func testActiveLettersAtTime() throws {
        let json = """
        {
          "word": "dog",
          "phonemes": [
            { "phoneme": "D",  "letters": [0], "start": 0.00, "end": 0.40 },
            { "phoneme": "AO", "letters": [1], "start": 0.40, "end": 0.95 },
            { "phoneme": "G",  "letters": [2], "start": 0.95, "end": 1.40 }
          ]
        }
        """.data(using: .utf8)!

        let timing = try JSONDecoder().decode(TimingData.self, from: json)
        XCTAssertEqual(timing.activeLetterIndices(at: 0.20), [0])
        XCTAssertEqual(timing.activeLetterIndices(at: 0.60), [1])
        XCTAssertEqual(timing.activeLetterIndices(at: 1.10), [2])
        XCTAssertTrue(timing.activeLetterIndices(at: 1.50).isEmpty) // past end
    }

    func testSpokenLetterIndicesAtTime() throws {
        let json = """
        {
          "word": "dog",
          "phonemes": [
            { "phoneme": "D",  "letters": [0], "start": 0.00, "end": 0.40 },
            { "phoneme": "AO", "letters": [1], "start": 0.40, "end": 0.95 },
            { "phoneme": "G",  "letters": [2], "start": 0.95, "end": 1.40 }
          ]
        }
        """.data(using: .utf8)!

        let timing = try JSONDecoder().decode(TimingData.self, from: json)
        // At time 0.60, "D" is spoken, "AO" is active, "G" is upcoming
        XCTAssertEqual(timing.spokenLetterIndices(at: 0.60), [0])
        // At time 1.10, "D" and "AO" are spoken, "G" is active
        XCTAssertEqual(timing.spokenLetterIndices(at: 1.10), [0, 1])
    }

    func testLoadFromBundle() {
        let timing = TimingData.load(word: "dog", from: Bundle.main)
        XCTAssertNotNil(timing)
        XCTAssertEqual(timing?.word, "dog")
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Expected: FAIL — `TimingData` type not found

- [ ] **Step 4: Implement TimingData**

```swift
// ARYA/Data/TimingData.swift
import Foundation

struct PhonemeTimingEntry: Codable {
    let phoneme: String
    let letters: [Int]
    let start: Double
    let end: Double
}

struct TimingData: Codable {
    let word: String
    let phonemes: [PhonemeTimingEntry]

    /// Returns letter indices currently being spoken at the given time.
    func activeLetterIndices(at time: Double) -> [Int] {
        for entry in phonemes {
            if time >= entry.start && time < entry.end {
                return entry.letters
            }
        }
        return []
    }

    /// Returns letter indices that have already been fully spoken at the given time.
    func spokenLetterIndices(at time: Double) -> [Int] {
        var spoken: [Int] = []
        for entry in phonemes {
            if entry.end <= time {
                spoken.append(contentsOf: entry.letters)
            }
        }
        return spoken
    }

    /// Total duration of the word audio based on timing data.
    var duration: Double {
        phonemes.last?.end ?? 0
    }

    static func load(word: String, from bundle: Bundle = .main) -> TimingData? {
        guard let url = bundle.url(forResource: "timing", withExtension: "json", subdirectory: "Vocabulary/\(word)"),
              let data = try? Data(contentsOf: url),
              let timing = try? JSONDecoder().decode(TimingData.self, from: data) else {
            return nil
        }
        return timing
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: add TimingData model with phoneme-to-letter mapping"
```

---

## Task 4: Detection Logic — ConsensusGate

**Files:**
- Create: `ARYA/Detection/ConsensusGate.swift`
- Test: `ARYATests/ConsensusGateTests.swift`

- [ ] **Step 1: Write failing tests**

```swift
// ARYATests/ConsensusGateTests.swift
import XCTest
@testable import ARYA

final class ConsensusGateTests: XCTestCase {

    let gate = ConsensusGate()

    func testBothAgreeHighConfidence_Accepts() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.90, vnSecondConfidence: 0.10,
            clipWord: "dog", clipSimilarity: 0.85, clipSecondSimilarity: 0.30
        )
        XCTAssertEqual(result, "dog")
    }

    func testDisagreement_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.90, vnSecondConfidence: 0.10,
            clipWord: "cat", clipSimilarity: 0.85, clipSecondSimilarity: 0.30
        )
        XCTAssertNil(result)
    }

    func testVNConfidenceTooLow_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.50, vnSecondConfidence: 0.10,
            clipWord: "dog", clipSimilarity: 0.85, clipSecondSimilarity: 0.30
        )
        XCTAssertNil(result)
    }

    func testCLIPSimilarityTooLow_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.90, vnSecondConfidence: 0.10,
            clipWord: "dog", clipSimilarity: 0.60, clipSecondSimilarity: 0.30
        )
        XCTAssertNil(result)
    }

    func testVNMarginTooSmall_Rejects() {
        // top-1 must be >= 1.5x top-2
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.80, vnSecondConfidence: 0.60,
            clipWord: "dog", clipSimilarity: 0.85, clipSecondSimilarity: 0.30
        )
        XCTAssertNil(result)
    }

    func testCLIPMarginTooSmall_Rejects() {
        // top-1 must be >= 1.3x top-2
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.90, vnSecondConfidence: 0.10,
            clipWord: "dog", clipSimilarity: 0.80, clipSecondSimilarity: 0.70
        )
        XCTAssertNil(result)
    }

    func testNilVNWord_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: nil, vnConfidence: 0.90, vnSecondConfidence: 0.10,
            clipWord: "dog", clipSimilarity: 0.85, clipSecondSimilarity: 0.30
        )
        XCTAssertNil(result)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — `ConsensusGate` type not found

- [ ] **Step 3: Implement ConsensusGate**

```swift
// ARYA/Detection/ConsensusGate.swift
import Foundation

struct ConsensusGate {
    let vnConfidenceThreshold: Double
    let vnMarginMultiplier: Double
    let clipSimilarityThreshold: Double
    let clipMarginMultiplier: Double

    init(
        vnConfidenceThreshold: Double = 0.70,
        vnMarginMultiplier: Double = 1.5,
        clipSimilarityThreshold: Double = 0.75,
        clipMarginMultiplier: Double = 1.3
    ) {
        self.vnConfidenceThreshold = vnConfidenceThreshold
        self.vnMarginMultiplier = vnMarginMultiplier
        self.clipSimilarityThreshold = clipSimilarityThreshold
        self.clipMarginMultiplier = clipMarginMultiplier
    }

    /// Returns the agreed-upon child word if both classifiers agree with sufficient confidence, nil otherwise.
    func evaluate(
        vnClassifyWord: String?,
        vnConfidence: Double,
        vnSecondConfidence: Double,
        clipWord: String,
        clipSimilarity: Double,
        clipSecondSimilarity: Double
    ) -> String? {
        // VN must have a mapped word
        guard let vnWord = vnClassifyWord else { return nil }

        // Both must agree
        guard vnWord == clipWord else { return nil }

        // VN confidence gate
        guard vnConfidence >= vnConfidenceThreshold else { return nil }

        // VN margin gate
        guard vnSecondConfidence == 0 || vnConfidence >= vnSecondConfidence * vnMarginMultiplier else { return nil }

        // CLIP similarity gate
        guard clipSimilarity >= clipSimilarityThreshold else { return nil }

        // CLIP margin gate
        guard clipSecondSimilarity == 0 || clipSimilarity >= clipSecondSimilarity * clipMarginMultiplier else { return nil }

        return vnWord
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add ConsensusGate with dual-model confidence thresholds"
```

---

## Task 5: Detection Logic — InstanceTracker

**Files:**
- Create: `ARYA/Detection/InstanceTracker.swift`
- Test: `ARYATests/InstanceTrackerTests.swift`

- [ ] **Step 1: Write failing tests**

```swift
// ARYATests/InstanceTrackerTests.swift
import XCTest
import CoreGraphics
@testable import ARYA

final class InstanceTrackerTests: XCTestCase {

    func testNewInstanceNotImmediatelyTappable() {
        var tracker = InstanceTracker(stabilityDuration: 0.5)
        tracker.update(instances: [
            DetectedInstance(id: 1, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2))
        ], timestamp: 0.0)

        let tappable = tracker.tappableInstances(at: 0.0)
        XCTAssertTrue(tappable.isEmpty)
    }

    func testInstanceBecomesTappableAfterStabilityDuration() {
        var tracker = InstanceTracker(stabilityDuration: 0.5)
        let instance = DetectedInstance(id: 1, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2))

        tracker.update(instances: [instance], timestamp: 0.0)
        tracker.update(instances: [instance], timestamp: 0.2)
        tracker.update(instances: [instance], timestamp: 0.5)

        let tappable = tracker.tappableInstances(at: 0.5)
        XCTAssertEqual(tappable.count, 1)
    }

    func testInstanceDisappearsAndResets() {
        var tracker = InstanceTracker(stabilityDuration: 0.5)
        let instance = DetectedInstance(id: 1, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2))

        tracker.update(instances: [instance], timestamp: 0.0)
        tracker.update(instances: [instance], timestamp: 0.3)
        tracker.update(instances: [], timestamp: 0.4) // disappeared
        tracker.update(instances: [instance], timestamp: 0.5) // reappeared

        let tappable = tracker.tappableInstances(at: 0.5)
        XCTAssertTrue(tappable.isEmpty) // timer reset
    }

    func testHitTestFindsCorrectInstance() {
        var tracker = InstanceTracker(stabilityDuration: 0.0) // no wait for this test
        let instanceA = DetectedInstance(id: 1, boundingBox: CGRect(x: 0.0, y: 0.0, width: 0.3, height: 0.3))
        let instanceB = DetectedInstance(id: 2, boundingBox: CGRect(x: 0.5, y: 0.5, width: 0.3, height: 0.3))

        tracker.update(instances: [instanceA, instanceB], timestamp: 0.0)

        let hit = tracker.instance(at: CGPoint(x: 0.15, y: 0.15))
        XCTAssertEqual(hit?.id, 1)

        let hitB = tracker.instance(at: CGPoint(x: 0.65, y: 0.65))
        XCTAssertEqual(hitB?.id, 2)

        let miss = tracker.instance(at: CGPoint(x: 0.9, y: 0.9))
        XCTAssertNil(miss)
    }

    func testIoUMatchingAcrossFrames() {
        var tracker = InstanceTracker(stabilityDuration: 0.5)
        // Frame 1: instance at position A
        tracker.update(instances: [
            DetectedInstance(id: 1, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2))
        ], timestamp: 0.0)
        // Frame 2: instance shifted slightly (same object, new ID from Vision)
        tracker.update(instances: [
            DetectedInstance(id: 5, boundingBox: CGRect(x: 0.12, y: 0.12, width: 0.2, height: 0.2))
        ], timestamp: 0.3)
        // Frame 3: same position
        tracker.update(instances: [
            DetectedInstance(id: 8, boundingBox: CGRect(x: 0.12, y: 0.12, width: 0.2, height: 0.2))
        ], timestamp: 0.5)

        // Should be tappable — tracker matched across frames via IoU
        let tappable = tracker.tappableInstances(at: 0.5)
        XCTAssertEqual(tappable.count, 1)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — types not found

- [ ] **Step 3: Implement InstanceTracker**

```swift
// ARYA/Detection/InstanceTracker.swift
import Foundation
import CoreGraphics

struct DetectedInstance: Equatable {
    let id: Int
    let boundingBox: CGRect // In normalized image coordinates (0-1)
}

struct TrackedInstance {
    var currentInstance: DetectedInstance
    var firstSeenTimestamp: Double
    var lastSeenTimestamp: Double
}

struct InstanceTracker {
    let stabilityDuration: Double
    private let iouThreshold: Double
    private(set) var trackedInstances: [TrackedInstance] = []

    init(stabilityDuration: Double = 0.5, iouThreshold: Double = 0.3) {
        self.stabilityDuration = stabilityDuration
        self.iouThreshold = iouThreshold
    }

    mutating func update(instances: [DetectedInstance], timestamp: Double) {
        var newTracked: [TrackedInstance] = []

        for instance in instances {
            if let matchIndex = bestMatch(for: instance) {
                // Existing instance — update
                var tracked = trackedInstances[matchIndex]
                tracked.currentInstance = instance
                tracked.lastSeenTimestamp = timestamp
                newTracked.append(tracked)
            } else {
                // New instance
                newTracked.append(TrackedInstance(
                    currentInstance: instance,
                    firstSeenTimestamp: timestamp,
                    lastSeenTimestamp: timestamp
                ))
            }
        }

        trackedInstances = newTracked
    }

    func tappableInstances(at currentTime: Double) -> [DetectedInstance] {
        trackedInstances
            .filter { (currentTime - $0.firstSeenTimestamp) >= stabilityDuration }
            .map(\.currentInstance)
    }

    func instance(at point: CGPoint) -> DetectedInstance? {
        trackedInstances
            .map(\.currentInstance)
            .first { $0.boundingBox.contains(point) }
    }

    private func bestMatch(for instance: DetectedInstance) -> Int? {
        var bestIndex: Int?
        var bestIoU: CGFloat = 0

        for (index, tracked) in trackedInstances.enumerated() {
            let iou = computeIoU(tracked.currentInstance.boundingBox, instance.boundingBox)
            if iou > CGFloat(iouThreshold) && iou > bestIoU {
                bestIoU = iou
                bestIndex = index
            }
        }

        return bestIndex
    }

    private func computeIoU(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        guard !intersection.isNull else { return 0 }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = (a.width * a.height) + (b.width * b.height) - intersectionArea
        guard unionArea > 0 else { return 0 }
        return intersectionArea / unionArea
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add InstanceTracker with IoU matching and stability timing"
```

---

## Task 6: Speech Logic — LetterHighlighter

**Files:**
- Create: `ARYA/Speech/LetterHighlighter.swift`
- Test: `ARYATests/LetterHighlighterTests.swift`

- [ ] **Step 1: Write failing tests**

```swift
// ARYATests/LetterHighlighterTests.swift
import XCTest
@testable import ARYA

final class LetterHighlighterTests: XCTestCase {

    func makeTimingData() -> TimingData {
        let json = """
        {
          "word": "dog",
          "phonemes": [
            { "phoneme": "D",  "letters": [0], "start": 0.00, "end": 0.40 },
            { "phoneme": "AO", "letters": [1], "start": 0.40, "end": 0.95 },
            { "phoneme": "G",  "letters": [2], "start": 0.95, "end": 1.40 }
          ]
        }
        """.data(using: .utf8)!
        return try! JSONDecoder().decode(TimingData.self, from: json)
    }

    func testAllLettersStartDim() {
        let highlighter = LetterHighlighter(timing: makeTimingData())
        let states = highlighter.letterStates(at: -0.1)
        XCTAssertEqual(states.count, 3)
        XCTAssertTrue(states.allSatisfy { $0 == .upcoming })
    }

    func testFirstLetterActiveAtStart() {
        let highlighter = LetterHighlighter(timing: makeTimingData())
        let states = highlighter.letterStates(at: 0.20)
        XCTAssertEqual(states[0], .active)
        XCTAssertEqual(states[1], .upcoming)
        XCTAssertEqual(states[2], .upcoming)
    }

    func testMiddleLetterActive() {
        let highlighter = LetterHighlighter(timing: makeTimingData())
        let states = highlighter.letterStates(at: 0.60)
        XCTAssertEqual(states[0], .spoken)
        XCTAssertEqual(states[1], .active)
        XCTAssertEqual(states[2], .upcoming)
    }

    func testAllSpokenAfterEnd() {
        let highlighter = LetterHighlighter(timing: makeTimingData())
        let states = highlighter.letterStates(at: 1.50)
        XCTAssertTrue(states.allSatisfy { $0 == .spoken })
    }

    func testMultiLetterPhoneme() {
        let json = """
        {
          "word": "elephant",
          "phonemes": [
            { "phoneme": "EH", "letters": [0], "start": 0.00, "end": 0.28 },
            { "phoneme": "L",  "letters": [1], "start": 0.28, "end": 0.52 },
            { "phoneme": "AH", "letters": [2], "start": 0.52, "end": 0.78 },
            { "phoneme": "F",  "letters": [3, 4], "start": 0.78, "end": 1.10 },
            { "phoneme": "AH", "letters": [5], "start": 1.10, "end": 1.35 },
            { "phoneme": "N",  "letters": [6], "start": 1.35, "end": 1.62 },
            { "phoneme": "T",  "letters": [7], "start": 1.62, "end": 1.85 }
          ]
        }
        """.data(using: .utf8)!
        let timing = try! JSONDecoder().decode(TimingData.self, from: json)
        let highlighter = LetterHighlighter(timing: timing)

        // At 0.90s, "ph" (indices 3 and 4) should both be active
        let states = highlighter.letterStates(at: 0.90)
        XCTAssertEqual(states[3], .active)
        XCTAssertEqual(states[4], .active)
        XCTAssertEqual(states[2], .spoken)
        XCTAssertEqual(states[5], .upcoming)
    }

    func testWordWithSpace() {
        let json = """
        {
          "word": "teddy bear",
          "phonemes": [
            { "phoneme": "T",  "letters": [0], "start": 0.00, "end": 0.20 },
            { "phoneme": "EH", "letters": [1], "start": 0.20, "end": 0.45 },
            { "phoneme": "D",  "letters": [2, 3], "start": 0.45, "end": 0.70 },
            { "phoneme": "IY", "letters": [4], "start": 0.70, "end": 1.00 },
            { "phoneme": "B",  "letters": [6], "start": 1.10, "end": 1.30 },
            { "phoneme": "EH", "letters": [7], "start": 1.30, "end": 1.55 },
            { "phoneme": "R",  "letters": [8, 9], "start": 1.55, "end": 1.85 }
          ]
        }
        """.data(using: .utf8)!
        let timing = try! JSONDecoder().decode(TimingData.self, from: json)
        let highlighter = LetterHighlighter(timing: timing)

        // "teddy bear" has 10 characters (index 5 is space)
        let states = highlighter.letterStates(at: 1.20)
        XCTAssertEqual(states.count, 10)
        XCTAssertEqual(states[5], .space) // space stays neutral
        XCTAssertEqual(states[6], .active)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — `LetterHighlighter` type not found

- [ ] **Step 3: Implement LetterHighlighter**

```swift
// ARYA/Speech/LetterHighlighter.swift
import Foundation

enum LetterState: Equatable {
    case upcoming  // Not yet spoken — dim white
    case active    // Currently being spoken — bright gold with glow
    case spoken    // Already spoken — softer gold
    case space     // Space character — always neutral
}

struct LetterHighlighter {
    let timing: TimingData
    let letterCount: Int

    init(timing: TimingData) {
        self.timing = timing
        self.letterCount = timing.word.count
    }

    func letterStates(at time: Double) -> [LetterState] {
        let activeIndices = Set(timing.activeLetterIndices(at: time))
        let spokenIndices = Set(timing.spokenLetterIndices(at: time))

        // All letter indices referenced in any phoneme
        let allReferencedIndices = Set(timing.phonemes.flatMap(\.letters))

        return (0..<letterCount).map { index in
            let char = timing.word[timing.word.index(timing.word.startIndex, offsetBy: index)]
            if char == " " {
                return .space
            }
            if activeIndices.contains(index) {
                return .active
            }
            if spokenIndices.contains(index) {
                return .spoken
            }
            // If past all phonemes, everything referenced is spoken
            if time >= (timing.phonemes.last?.end ?? 0) && allReferencedIndices.contains(index) {
                return .spoken
            }
            return .upcoming
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add LetterHighlighter with per-letter state tracking"
```

---

## Task 7: Camera Setup

**Files:**
- Create: `ARYA/Detection/CameraManager.swift`
- Create: `ARYA/Views/CameraPreviewView.swift`

- [ ] **Step 1: Implement CameraManager**

```swift
// ARYA/Detection/CameraManager.swift
import AVFoundation
import UIKit

protocol CameraManagerDelegate: AnyObject {
    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime)
}

final class CameraManager: NSObject {
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.arya.camera.session")
    private let outputQueue = DispatchQueue(label: "com.arya.camera.output")

    weak var delegate: CameraManagerDelegate?
    private var frameCount = 0

    var previewLayer: AVCaptureVideoPreviewLayer {
        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        layer.videoGravity = .resizeAspectFill
        return layer
    }

    func configure() {
        sessionQueue.async { [weak self] in
            self?.setupSession()
        }
    }

    func start() {
        sessionQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
        }
    }

    private func setupSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1920x1080

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera),
              captureSession.canAddInput(input) else {
            captureSession.commitConfiguration()
            return
        }

        captureSession.addInput(input)

        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        guard captureSession.canAddOutput(videoOutput) else {
            captureSession.commitConfiguration()
            return
        }

        captureSession.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            connection.videoRotationAngle = 90 // Portrait
        }

        captureSession.commitConfiguration()
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameCount += 1
        // Process every 3rd frame for segmentation (~10fps on 30fps feed)
        guard frameCount % 3 == 0 else { return }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        delegate?.cameraManager(self, didOutput: pixelBuffer, timestamp: timestamp)
    }
}
```

- [ ] **Step 2: Implement CameraPreviewView**

```swift
// ARYA/Views/CameraPreviewView.swift
import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let previewLayer = cameraManager.previewLayer
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        if let previewLayer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            previewLayer.frame = uiView.bounds
        }
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add CameraManager and CameraPreviewView"
```

---

## Task 8: Segmentation Engine

**Files:**
- Create: `ARYA/Detection/SegmentationEngine.swift`

- [ ] **Step 1: Implement SegmentationEngine**

```swift
// ARYA/Detection/SegmentationEngine.swift
import Vision
import CoreImage
import UIKit

struct SegmentationResult {
    let instances: [DetectedInstance]
    let observation: VNInstanceMaskObservation?
    let pixelBuffer: CVPixelBuffer
}

final class SegmentationEngine {
    private let request = VNGenerateForegroundInstanceMaskRequest()

    func segment(pixelBuffer: CVPixelBuffer) -> SegmentationResult {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        do {
            try handler.perform([request])
        } catch {
            return SegmentationResult(instances: [], observation: nil, pixelBuffer: pixelBuffer)
        }

        guard let observation = request.results?.first else {
            return SegmentationResult(instances: [], observation: nil, pixelBuffer: pixelBuffer)
        }

        let allInstances = observation.allInstances
        var detected: [DetectedInstance] = []

        for index in allInstances {
            if let mask = try? observation.generateScaledMaskForImage(forInstances: IndexSet(integer: index), from: handler) {
                let boundingBox = computeBoundingBox(from: mask)
                detected.append(DetectedInstance(id: index, boundingBox: boundingBox))
            }
        }

        return SegmentationResult(instances: detected, observation: observation, pixelBuffer: pixelBuffer)
    }

    /// Generate a CIImage mask for a specific instance.
    func maskImage(for instanceIndex: Int, observation: VNInstanceMaskObservation, handler: VNImageRequestHandler) -> CIImage? {
        guard let mask = try? observation.generateScaledMaskForImage(forInstances: IndexSet(integer: instanceIndex), from: handler) else {
            return nil
        }
        return CIImage(cvPixelBuffer: mask)
    }

    private func computeBoundingBox(from maskBuffer: CVPixelBuffer) -> CGRect {
        let width = CVPixelBufferGetWidth(maskBuffer)
        let height = CVPixelBufferGetHeight(maskBuffer)

        CVPixelBufferLockBaseAddress(maskBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(maskBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(maskBuffer) else {
            return .zero
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(maskBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)

        var minX = width, minY = height, maxX = 0, maxY = 0

        for y in 0..<height {
            for x in 0..<width {
                let pixel = buffer[y * bytesPerRow + x]
                if pixel > 128 {
                    minX = min(minX, x)
                    minY = min(minY, y)
                    maxX = max(maxX, x)
                    maxY = max(maxY, y)
                }
            }
        }

        guard maxX > minX && maxY > minY else { return .zero }

        // Normalize to 0-1
        return CGRect(
            x: CGFloat(minX) / CGFloat(width),
            y: CGFloat(minY) / CGFloat(height),
            width: CGFloat(maxX - minX) / CGFloat(width),
            height: CGFloat(maxY - minY) / CGFloat(height)
        )
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "feat: add SegmentationEngine with VNGenerateForegroundInstanceMaskRequest"
```

---

## Task 9: Classification Engine

**Files:**
- Create: `ARYA/Detection/ClassificationEngine.swift`
- Create: `ARYA/Data/CLIPEmbeddings.swift`

This task requires the MobileCLIP CoreML model. For initial development, the VNClassify path works without it. MobileCLIP integration will use a placeholder until the model is downloaded in Task 13.

- [ ] **Step 1: Implement CLIPEmbeddings (with placeholder for model)**

```swift
// ARYA/Data/CLIPEmbeddings.swift
import Foundation
import CoreML
import Accelerate

struct CLIPResult {
    let word: String
    let similarity: Double
}

final class CLIPEmbeddings {
    private let vocabulary: [String]
    private let embeddings: [[Float]] // One embedding vector per vocabulary word
    private var imageEncoder: MLModel?

    init(vocabulary: [String], embeddings: [[Float]], imageEncoder: MLModel? = nil) {
        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.imageEncoder = imageEncoder
    }

    /// Load pre-computed text embeddings from binary file.
    static func load(vocabulary: [String], from bundle: Bundle = .main) -> CLIPEmbeddings? {
        guard let url = bundle.url(forResource: "text_embeddings", withExtension: "bin"),
              let data = try? Data(contentsOf: url) else {
            return nil
        }

        let embeddingDim = 512 // MobileCLIP S2 dimension
        let floatCount = data.count / MemoryLayout<Float>.size
        let vectorCount = floatCount / embeddingDim

        guard vectorCount == vocabulary.count else { return nil }

        var allFloats = [Float](repeating: 0, count: floatCount)
        data.withUnsafeBytes { ptr in
            allFloats = Array(ptr.bindMemory(to: Float.self))
        }

        var embeddings: [[Float]] = []
        for i in 0..<vectorCount {
            let start = i * embeddingDim
            let vector = Array(allFloats[start..<start + embeddingDim])
            embeddings.append(vector)
        }

        // Try to load the MobileCLIP image encoder model
        let modelURL = bundle.url(forResource: "MobileCLIPImageEncoder", withExtension: "mlmodelc")
        let model = modelURL.flatMap { try? MLModel(contentsOf: $0) }

        return CLIPEmbeddings(vocabulary: vocabulary, embeddings: embeddings, imageEncoder: model)
    }

    /// Classify a cropped image against the vocabulary. Returns top-2 results.
    func classify(imageBuffer: CVPixelBuffer) -> (top1: CLIPResult, top2: CLIPResult)? {
        guard let encoder = imageEncoder else { return nil }

        // Run image through MobileCLIP image encoder
        guard let imageEmbedding = encodeImage(imageBuffer, with: encoder) else { return nil }

        // Compute cosine similarity against all vocabulary embeddings
        var similarities: [(word: String, similarity: Double)] = []
        for (index, textEmbedding) in embeddings.enumerated() {
            let sim = cosineSimilarity(imageEmbedding, textEmbedding)
            similarities.append((vocabulary[index], Double(sim)))
        }

        similarities.sort { $0.similarity > $1.similarity }

        guard similarities.count >= 2 else { return nil }

        return (
            top1: CLIPResult(word: similarities[0].word, similarity: similarities[0].similarity),
            top2: CLIPResult(word: similarities[1].word, similarity: similarities[1].similarity)
        )
    }

    private func encodeImage(_ buffer: CVPixelBuffer, with model: MLModel) -> [Float]? {
        // Create MLFeatureValue from pixel buffer
        guard let input = try? MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)]),
              let output = try? model.prediction(from: input),
              let embeddingFeature = output.featureValue(for: "embedding"),
              let multiArray = embeddingFeature.multiArrayValue else {
            return nil
        }

        let count = multiArray.count
        var result = [Float](repeating: 0, count: count)
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            result[i] = ptr[i]
        }
        return result
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
        vDSP_dotpr(a, 1, a, 1, &normA, vDSP_Length(a.count))
        vDSP_dotpr(b, 1, b, 1, &normB, vDSP_Length(b.count))
        guard normA > 0 && normB > 0 else { return 0 }
        return dotProduct / (sqrt(normA) * sqrt(normB))
    }
}
```

- [ ] **Step 2: Implement ClassificationEngine**

```swift
// ARYA/Detection/ClassificationEngine.swift
import Vision
import CoreImage
import UIKit

struct ClassificationResult {
    let word: String
}

final class ClassificationEngine {
    private let labelMapper: LabelMapper
    private let clipEmbeddings: CLIPEmbeddings?
    private let consensusGate: ConsensusGate

    init(labelMapper: LabelMapper, clipEmbeddings: CLIPEmbeddings?, consensusGate: ConsensusGate = ConsensusGate()) {
        self.labelMapper = labelMapper
        self.clipEmbeddings = clipEmbeddings
        self.consensusGate = consensusGate
    }

    /// Classify a cropped object image. Returns the child-friendly word or nil.
    func classify(imageBuffer: CVPixelBuffer) async -> ClassificationResult? {
        async let vnResult = runVNClassify(imageBuffer)
        async let clipResult = runCLIP(imageBuffer)

        let vn = await vnResult
        let clip = await clipResult

        guard let vn = vn else { return nil }

        // Map VN label to child word
        let vnChildWord = labelMapper.childWord(for: vn.label)

        // If no CLIP available, use VN alone with stricter threshold
        guard let clip = clip else {
            // Fallback: VN only with very high confidence
            guard let word = vnChildWord,
                  vn.confidence >= 0.85,
                  vn.secondConfidence == 0 || vn.confidence >= vn.secondConfidence * 2.0 else {
                return nil
            }
            return ClassificationResult(word: word)
        }

        // Full dual-model consensus
        guard let word = consensusGate.evaluate(
            vnClassifyWord: vnChildWord,
            vnConfidence: Double(vn.confidence),
            vnSecondConfidence: Double(vn.secondConfidence),
            clipWord: clip.top1.word,
            clipSimilarity: clip.top1.similarity,
            clipSecondSimilarity: clip.top2.similarity
        ) else {
            return nil
        }

        return ClassificationResult(word: word)
    }

    private struct VNResult {
        let label: String
        let confidence: Float
        let secondConfidence: Float
    }

    private func runVNClassify(_ buffer: CVPixelBuffer) async -> VNResult? {
        await withCheckedContinuation { continuation in
            let request = VNClassifyImageRequest { request, error in
                guard let observations = request.results as? [VNClassificationObservation],
                      observations.count >= 2 else {
                    continuation.resume(returning: nil)
                    return
                }
                continuation.resume(returning: VNResult(
                    label: observations[0].identifier,
                    confidence: observations[0].confidence,
                    secondConfidence: observations[1].confidence
                ))
            }

            let handler = VNImageRequestHandler(cvPixelBuffer: buffer, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(returning: nil)
            }
        }
    }

    private func runCLIP(_ buffer: CVPixelBuffer) async -> (top1: CLIPResult, top2: CLIPResult)? {
        clipEmbeddings?.classify(imageBuffer: buffer)
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add ClassificationEngine with dual VNClassify + MobileCLIP pipeline"
```

---

## Task 10: Speech — WordSpeaker

**Files:**
- Create: `ARYA/Speech/WordSpeaker.swift`

- [ ] **Step 1: Implement WordSpeaker**

```swift
// ARYA/Speech/WordSpeaker.swift
import AVFoundation
import Combine

final class WordSpeaker: ObservableObject {
    @Published private(set) var currentTime: Double = 0
    @Published private(set) var isPlaying: Bool = false

    private var audioPlayer: AVAudioPlayer?
    private var displayLink: CADisplayLink?
    private var onComplete: (() -> Void)?

    func speak(word: String, from bundle: Bundle = .main, onComplete: @escaping () -> Void) {
        stop()
        self.onComplete = onComplete

        guard let url = bundle.url(forResource: "audio", withExtension: "m4a", subdirectory: "Vocabulary/\(word)"),
              let player = try? AVAudioPlayer(contentsOf: url) else {
            onComplete()
            return
        }

        // Configure audio session for playback
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)

        audioPlayer = player
        player.delegate = self
        player.prepareToPlay()
        player.play()
        isPlaying = true

        startDisplayLink()
    }

    func stop() {
        audioPlayer?.stop()
        audioPlayer = nil
        stopDisplayLink()
        currentTime = 0
        isPlaying = false
    }

    private func startDisplayLink() {
        stopDisplayLink()
        let link = CADisplayLink(target: self, selector: #selector(displayLinkFired))
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    private func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc private func displayLinkFired() {
        guard let player = audioPlayer else { return }
        currentTime = player.currentTime
    }
}

extension WordSpeaker: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        // Keep final time for "all letters glow" state
        currentTime = player.duration
        isPlaying = false
        stopDisplayLink()

        // Delay before calling completion to show all-glow state
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.onComplete?()
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "feat: add WordSpeaker with CADisplayLink-driven timing"
```

---

## Task 11: App State & Core Views

**Files:**
- Create: `ARYA/App/AppState.swift`
- Create: `ARYA/Views/ContentView.swift`
- Create: `ARYA/Views/OnboardingHintView.swift`
- Modify: `ARYA/App/ARYAApp.swift`

- [ ] **Step 1: Implement AppState**

```swift
// ARYA/App/AppState.swift
import SwiftUI
import Vision

enum AppMode: Equatable {
    case exploring
    case classifying // Brief transition while classification runs
    case learning(word: String, instanceIndex: Int)
}

@MainActor
final class AppState: ObservableObject {
    @Published var mode: AppMode = .exploring
    @Published private(set) var hasCompletedFirstTap: Bool

    let cameraManager = CameraManager()
    let segmentationEngine = SegmentationEngine()
    let wordSpeaker = WordSpeaker()
    let vocabularyStore: VocabularyStore
    let labelMapper: LabelMapper
    let classificationEngine: ClassificationEngine
    var instanceTracker = InstanceTracker()

    // Latest segmentation result for mask access
    @Published var latestSegmentation: SegmentationResult?

    init() {
        hasCompletedFirstTap = UserDefaults.standard.bool(forKey: "hasCompletedFirstTap")
        vocabularyStore = VocabularyStore.load()
        labelMapper = LabelMapper.load()

        let clipEmbeddings = CLIPEmbeddings.load(vocabulary: vocabularyStore.entries.map(\.word))
        classificationEngine = ClassificationEngine(labelMapper: labelMapper, clipEmbeddings: clipEmbeddings)
    }

    func handleTap(at normalizedPoint: CGPoint) {
        guard mode == .exploring else { return }

        // Check if tap hit a tappable instance
        guard let instance = instanceTracker.instance(at: normalizedPoint) else { return }

        // Trigger haptic
        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()

        mode = .classifying

        guard let segResult = latestSegmentation else {
            mode = .exploring
            return
        }

        Task {
            // Crop the instance region from the full frame
            let croppedBuffer = cropPixelBuffer(segResult.pixelBuffer, to: instance.boundingBox)

            guard let croppedBuffer = croppedBuffer,
                  let result = await classificationEngine.classify(imageBuffer: croppedBuffer) else {
                mode = .exploring
                return
            }

            // Mark first tap complete
            if !hasCompletedFirstTap {
                hasCompletedFirstTap = true
                UserDefaults.standard.set(true, forKey: "hasCompletedFirstTap")
            }

            mode = .learning(word: result.word, instanceIndex: instance.id)
        }
    }

    func dismissLearning() {
        wordSpeaker.stop()
        mode = .exploring
    }

    func processFrame(_ pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        let result = segmentationEngine.segment(pixelBuffer: pixelBuffer)
        let timeSeconds = CMTimeGetSeconds(timestamp)
        instanceTracker.update(instances: result.instances, timestamp: timeSeconds)
        latestSegmentation = result
    }

    private func cropPixelBuffer(_ buffer: CVPixelBuffer, to normalizedRect: CGRect) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)

        let cropRect = CGRect(
            x: normalizedRect.origin.x * CGFloat(width),
            y: normalizedRect.origin.y * CGFloat(height),
            width: normalizedRect.width * CGFloat(width),
            height: normalizedRect.height * CGFloat(height)
        ).integral

        guard cropRect.width > 0, cropRect.height > 0 else { return nil }

        let ciImage = CIImage(cvPixelBuffer: buffer).cropped(to: cropRect)
        let context = CIContext()
        var croppedBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(cropRect.width), Int(cropRect.height),
                           kCVPixelFormatType_32BGRA, nil, &croppedBuffer)
        guard let output = croppedBuffer else { return nil }
        context.render(ciImage, to: output)
        return output
    }
}
```

- [ ] **Step 2: Implement OnboardingHintView**

```swift
// ARYA/Views/OnboardingHintView.swift
import SwiftUI

struct OnboardingHintView: View {
    @State private var isPulsing = false

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "hand.tap.fill")
                .font(.system(size: 48))
                .foregroundColor(.white.opacity(0.7))
                .scaleEffect(isPulsing ? 1.1 : 0.95)
                .animation(
                    .easeInOut(duration: 1.0).repeatForever(autoreverses: true),
                    value: isPulsing
                )

            Text("tap")
                .font(.system(size: 16, weight: .medium, design: .rounded))
                .foregroundColor(.white.opacity(0.5))
        }
        .onAppear { isPulsing = true }
    }
}
```

- [ ] **Step 3: Implement ContentView**

```swift
// ARYA/Views/ContentView.swift
import SwiftUI

struct ContentView: View {
    @StateObject private var appState = AppState()

    var body: some View {
        ZStack {
            // Camera feed — always visible
            CameraPreviewView(cameraManager: appState.cameraManager)
                .ignoresSafeArea()
                .onTapGesture { location in
                    let screenSize = UIScreen.main.bounds.size
                    let normalized = CGPoint(
                        x: location.x / screenSize.width,
                        y: location.y / screenSize.height
                    )
                    if case .learning = appState.mode {
                        appState.dismissLearning()
                    } else {
                        appState.handleTap(at: normalized)
                    }
                }

            // Learning overlay
            if case .learning(let word, let instanceIndex) = appState.mode {
                LearningOverlayView(
                    word: word,
                    instanceIndex: instanceIndex,
                    segmentation: appState.latestSegmentation,
                    wordSpeaker: appState.wordSpeaker
                )
                .transition(.opacity.animation(.easeInOut(duration: 0.3)))
            }

            // First-launch hint
            if !appState.hasCompletedFirstTap && appState.mode == .exploring {
                OnboardingHintView()
            }
        }
        .onAppear {
            appState.cameraManager.configure()
            appState.cameraManager.start()
        }
        .preferredColorScheme(.dark)
        .statusBarHidden()
    }
}
```

- [ ] **Step 4: Update ARYAApp.swift**

```swift
// ARYA/App/ARYAApp.swift
import SwiftUI

@main
struct ARYAApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add AppState, ContentView, and OnboardingHintView"
```

---

## Task 12: Learning Overlay & Word Display Views

**Files:**
- Create: `ARYA/Views/LearningOverlayView.swift`
- Create: `ARYA/Views/WordDisplayView.swift`

- [ ] **Step 1: Implement WordDisplayView**

```swift
// ARYA/Views/WordDisplayView.swift
import SwiftUI

struct WordDisplayView: View {
    let word: String
    let letterStates: [LetterState]

    var body: some View {
        HStack(spacing: 16) {
            ForEach(Array(word.enumerated()), id: \.offset) { index, char in
                if char == " " {
                    Spacer().frame(width: 24)
                } else {
                    Text(String(char))
                        .font(.system(size: 80, weight: .bold, design: .rounded))
                        .foregroundColor(color(for: letterState(at: index)))
                        .shadow(color: glowColor(for: letterState(at: index)), radius: 20)
                        .animation(.easeInOut(duration: 0.15), value: letterState(at: index))
                }
            }
        }
    }

    private func letterState(at index: Int) -> LetterState {
        guard index < letterStates.count else { return .upcoming }
        return letterStates[index]
    }

    private func color(for state: LetterState) -> Color {
        switch state {
        case .upcoming: return .white.opacity(0.3)
        case .active:   return Color(red: 1.0, green: 0.85, blue: 0.24) // #FFD93D
        case .spoken:   return Color(red: 1.0, green: 0.85, blue: 0.24).opacity(0.45)
        case .space:    return .clear
        }
    }

    private func glowColor(for state: LetterState) -> Color {
        switch state {
        case .active: return Color(red: 1.0, green: 0.85, blue: 0.24).opacity(0.6)
        default:      return .clear
        }
    }
}
```

- [ ] **Step 2: Implement LearningOverlayView**

```swift
// ARYA/Views/LearningOverlayView.swift
import SwiftUI

struct LearningOverlayView: View {
    let word: String
    let instanceIndex: Int
    let segmentation: SegmentationResult?
    @ObservedObject var wordSpeaker: WordSpeaker

    @State private var letterHighlighter: LetterHighlighter?
    @State private var timingData: TimingData?
    @State private var hasStartedSpeaking = false

    var body: some View {
        ZStack {
            // Dim overlay
            Color.black.opacity(0.6)
                .ignoresSafeArea()

            // TODO: Task 15 will add the actual mask-based glow overlay here.
            // For now, the dim effect alone provides visual focus.

            // Word display
            VStack {
                Spacer()
                    .frame(height: UIScreen.main.bounds.height * 0.3)

                if let highlighter = letterHighlighter {
                    WordDisplayView(
                        word: word,
                        letterStates: highlighter.letterStates(at: wordSpeaker.currentTime)
                    )
                } else {
                    // Fallback: show word without animation
                    Text(word)
                        .font(.system(size: 80, weight: .bold, design: .rounded))
                        .foregroundColor(Color(red: 1.0, green: 0.85, blue: 0.24))
                }

                Spacer()
            }
        }
        .onAppear {
            // Load timing data and start speaking
            if let timing = TimingData.load(word: word) {
                timingData = timing
                letterHighlighter = LetterHighlighter(timing: timing)
            }

            if !hasStartedSpeaking {
                hasStartedSpeaking = true
                wordSpeaker.speak(word: word) {
                    // Audio complete — stay in learning mode until child taps
                }
            }
        }
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add LearningOverlayView and WordDisplayView with letter animation"
```

---

## Task 13: Wire Camera Frames to Segmentation

**Files:**
- Modify: `ARYA/App/AppState.swift`
- Modify: `ARYA/Views/ContentView.swift`

- [ ] **Step 1: Make AppState conform to CameraManagerDelegate**

Add to `AppState.swift`:

```swift
extension AppState: CameraManagerDelegate {
    nonisolated func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        // Run segmentation off main thread, then update state on main
        let result = segmentationEngine.segment(pixelBuffer: pixelBuffer)
        let timeSeconds = CMTimeGetSeconds(timestamp)

        Task { @MainActor in
            instanceTracker.update(instances: result.instances, timestamp: timeSeconds)
            latestSegmentation = result
        }
    }
}
```

- [ ] **Step 2: Set delegate in AppState.init**

Add to the `init()` method of `AppState`:

```swift
cameraManager.delegate = self
```

- [ ] **Step 3: Build and verify on simulator (segmentation won't work but app should launch)**

```bash
xcodebuild build -project ARYA.xcodeproj -scheme ARYA -destination 'platform=iOS Simulator,name=iPhone 16'
```

Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: wire camera frames to segmentation engine"
```

---

## Task 14: Build-Time Audio Pipeline (Python Scripts)

**Files:**
- Create: `scripts/requirements.txt`
- Create: `scripts/generate_audio.py`
- Create: `scripts/align_audio.py`
- Create: `scripts/build_timing_data.py`
- Create: `scripts/phoneme_to_letter_mappings.json`

- [ ] **Step 1: Create requirements.txt**

```
elevenlabs>=1.0.0
montreal-forced-aligner>=3.0.0
textgrid>=1.5
```

- [ ] **Step 2: Create generate_audio.py**

```python
#!/usr/bin/env python3
"""Generate audio files for each vocabulary word using ElevenLabs."""

import json
import os
import sys
from pathlib import Path

from elevenlabs import ElevenLabs

VOCAB_PATH = Path(__file__).parent.parent / "ARYA" / "Resources" / "vocabulary.json"
OUTPUT_DIR = Path(__file__).parent.parent / "ARYA" / "Resources" / "Vocabulary"

def main():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: Set ELEVENLABS_API_KEY environment variable")
        sys.exit(1)

    client = ElevenLabs(api_key=api_key)

    with open(VOCAB_PATH) as f:
        vocabulary = json.load(f)

    # Use a warm, friendly voice. "Rachel" is a good default.
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel

    for entry in vocabulary:
        word = entry["word"]
        word_dir = OUTPUT_DIR / word
        word_dir.mkdir(parents=True, exist_ok=True)

        audio_path = word_dir / "audio.m4a"
        if audio_path.exists():
            print(f"  Skipping {word} (already exists)")
            continue

        print(f"  Generating: {word}")

        audio_generator = client.text_to_speech.convert(
            text=word,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.75,
                "similarity_boost": 0.75,
                "speed": 0.7,
            },
        )

        # Save as mp3 first, then convert to m4a
        mp3_path = word_dir / "audio.mp3"
        with open(mp3_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)

        # Convert mp3 to m4a using ffmpeg
        os.system(f'ffmpeg -i "{mp3_path}" -c:a aac -b:a 128k "{audio_path}" -y -loglevel quiet')
        mp3_path.unlink()

        print(f"  Done: {word}")

    print(f"\nGenerated audio for {len(vocabulary)} words")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create align_audio.py**

```python
#!/usr/bin/env python3
"""Run Montreal Forced Aligner on generated audio files."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

VOCAB_PATH = Path(__file__).parent.parent / "ARYA" / "Resources" / "vocabulary.json"
AUDIO_DIR = Path(__file__).parent.parent / "ARYA" / "Resources" / "Vocabulary"

def main():
    with open(VOCAB_PATH) as f:
        vocabulary = json.load(f)

    for entry in vocabulary:
        word = entry["word"]
        audio_path = AUDIO_DIR / word / "audio.m4a"

        if not audio_path.exists():
            print(f"  Skipping {word} (no audio file)")
            continue

        # MFA needs a .txt file with the transcript alongside the audio
        # Create a temporary directory with the audio and transcript
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Convert m4a to wav for MFA
            wav_path = tmp / f"{word}.wav"
            os.system(f'ffmpeg -i "{audio_path}" -ar 16000 -ac 1 "{wav_path}" -y -loglevel quiet')

            # Create transcript file
            txt_path = tmp / f"{word}.txt"
            txt_path.write_text(word)

            # Output directory
            output_dir = tmp / "output"
            output_dir.mkdir()

            # Run MFA
            result = subprocess.run(
                [
                    "mfa", "align",
                    str(tmp),
                    "english_us_arpa",
                    "english_us_arpa",
                    str(output_dir),
                    "--clean",
                    "--single_speaker",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"  MFA failed for {word}: {result.stderr}")
                continue

            # Copy TextGrid to audio directory
            textgrid_path = output_dir / f"{word}.TextGrid"
            if textgrid_path.exists():
                dest = AUDIO_DIR / word / "aligned.TextGrid"
                dest.write_text(textgrid_path.read_text())
                print(f"  Aligned: {word}")
            else:
                print(f"  No TextGrid output for {word}")

    print("\nAlignment complete")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create phoneme_to_letter_mappings.json (starter set)**

```json
{
  "dog": [
    { "phoneme": "D", "letters": [0] },
    { "phoneme": "AO", "letters": [1] },
    { "phoneme": "G", "letters": [2] }
  ],
  "car": [
    { "phoneme": "K", "letters": [0] },
    { "phoneme": "AA", "letters": [1] },
    { "phoneme": "R", "letters": [2] }
  ],
  "cup": [
    { "phoneme": "K", "letters": [0] },
    { "phoneme": "AH", "letters": [1] },
    { "phoneme": "P", "letters": [2] }
  ],
  "chair": [
    { "phoneme": "CH", "letters": [0, 1] },
    { "phoneme": "EH", "letters": [2] },
    { "phoneme": "R", "letters": [3, 4] }
  ],
  "apple": [
    { "phoneme": "AE", "letters": [0] },
    { "phoneme": "P", "letters": [1, 2] },
    { "phoneme": "AH", "letters": [3] },
    { "phoneme": "L", "letters": [4] }
  ]
}
```

- [ ] **Step 5: Create build_timing_data.py**

```python
#!/usr/bin/env python3
"""Convert MFA TextGrid output + phoneme-to-letter mappings into timing.json files."""

import json
from pathlib import Path

try:
    import textgrid
except ImportError:
    print("Install textgrid: pip install textgrid")
    raise

AUDIO_DIR = Path(__file__).parent.parent / "ARYA" / "Resources" / "Vocabulary"
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
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: add Python build-time audio pipeline (ElevenLabs + MFA + timing)"
```

---

## Task 15: Mask-Based Object Highlighting

**Files:**
- Modify: `ARYA/Views/LearningOverlayView.swift`
- Create: `ARYA/Views/MaskOverlayView.swift`

- [ ] **Step 1: Implement MaskOverlayView**

```swift
// ARYA/Views/MaskOverlayView.swift
import SwiftUI
import Vision
import CoreImage

struct MaskOverlayView: UIViewRepresentable {
    let observation: VNInstanceMaskObservation?
    let instanceIndex: Int
    let pixelBuffer: CVPixelBuffer?

    func makeUIView(context: Context) -> MaskUIView {
        MaskUIView()
    }

    func updateUIView(_ uiView: MaskUIView, context: Context) {
        uiView.updateMask(observation: observation, instanceIndex: instanceIndex, pixelBuffer: pixelBuffer)
    }
}

class MaskUIView: UIView {
    private let dimLayer = CALayer()
    private let glowLayer = CALayer()
    private let ciContext = CIContext()

    override init(frame: CGRect) {
        super.init(frame: frame)
        isUserInteractionEnabled = false
        backgroundColor = .clear
        layer.addSublayer(dimLayer)
        layer.addSublayer(glowLayer)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func layoutSubviews() {
        super.layoutSubviews()
        dimLayer.frame = bounds
        glowLayer.frame = bounds
    }

    func updateMask(observation: VNInstanceMaskObservation?, instanceIndex: Int, pixelBuffer: CVPixelBuffer?) {
        guard let observation = observation, let pixelBuffer = pixelBuffer else {
            dimLayer.contents = nil
            glowLayer.contents = nil
            return
        }

        guard let maskBuffer = try? observation.generateScaledMaskForImage(
            forInstances: IndexSet(integer: instanceIndex),
            from: VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        ) else { return }

        let maskCI = CIImage(cvPixelBuffer: maskBuffer)
        let viewSize = bounds.size

        // Dim layer: invert mask (everything EXCEPT the object is dimmed)
        let invertedMask = maskCI.applyingFilter("CIColorInvert")
        let dimImage = invertedMask.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0.6),
        ])

        if let cgDim = ciContext.createCGImage(dimImage, from: dimImage.extent) {
            dimLayer.contents = cgDim
            dimLayer.contentsGravity = .resizeAspectFill
        }

        // Glow layer: blur the mask edges for a gold glow effect
        let goldMask = maskCI.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 1.0, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: 0.85, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0.24, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0.8),
        ])
        let blurredGlow = goldMask.applyingFilter("CIGaussianBlur", parameters: ["inputRadius": 12.0])

        if let cgGlow = ciContext.createCGImage(blurredGlow, from: blurredGlow.extent) {
            glowLayer.contents = cgGlow
            glowLayer.contentsGravity = .resizeAspectFill
        }
    }
}
```

- [ ] **Step 2: Update LearningOverlayView to use MaskOverlayView**

Replace the `// TODO` comment in `LearningOverlayView.swift` with:

```swift
// Mask-based highlighting
if let segResult = segmentation {
    MaskOverlayView(
        observation: segResult.observation,
        instanceIndex: instanceIndex,
        pixelBuffer: segResult.pixelBuffer
    )
    .ignoresSafeArea()
}
```

And remove the plain `Color.black.opacity(0.6)` dim overlay since the mask handles it.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add mask-based object highlighting with gold glow effect"
```

---

## Task 16: MobileCLIP Model Integration

**Files:**
- Create: `scripts/build_clip_embeddings.py`
- Create: `scripts/download_mobileclip.sh`

- [ ] **Step 1: Create download script for MobileCLIP CoreML model**

```bash
#!/usr/bin/env bash
# scripts/download_mobileclip.sh
# Downloads MobileCLIP S2 CoreML model from Hugging Face

set -e

MODEL_DIR="$(dirname "$0")/../ARYA/Resources"
mkdir -p "$MODEL_DIR"

echo "Downloading MobileCLIP S2 image encoder (CoreML)..."

# Clone just the CoreML model files from Hugging Face
pip install huggingface_hub 2>/dev/null

python3 -c "
from huggingface_hub import hf_hub_download
import shutil

# Download the image encoder CoreML model
path = hf_hub_download(
    repo_id='apple/coreml-mobileclip',
    filename='MobileCLIP-S2-ImageEncoder.mlpackage.zip',
    local_dir='$MODEL_DIR/tmp'
)
print(f'Downloaded to: {path}')
"

# Unzip and place model
cd "$MODEL_DIR/tmp"
unzip -o MobileCLIP-S2-ImageEncoder.mlpackage.zip -d "$MODEL_DIR/"
rm -rf "$MODEL_DIR/tmp"

echo "MobileCLIP model ready at $MODEL_DIR/MobileCLIPImageEncoder.mlpackage"
```

- [ ] **Step 2: Create build_clip_embeddings.py**

```python
#!/usr/bin/env python3
"""Pre-compute MobileCLIP text embeddings for vocabulary words."""

import json
import struct
from pathlib import Path

import torch
import mobileclip

VOCAB_PATH = Path(__file__).parent.parent / "ARYA" / "Resources" / "vocabulary.json"
OUTPUT_PATH = Path(__file__).parent.parent / "ARYA" / "Resources" / "text_embeddings.bin"

def main():
    with open(VOCAB_PATH) as f:
        vocabulary = json.load(f)

    words = [entry["word"] for entry in vocabulary]
    prompts = [f"a photo of a {word}" for word in words]

    # Load MobileCLIP S2 model
    model, _, preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s2", pretrained="checkpoints/mobileclip_s2.pt"
    )
    tokenizer = mobileclip.get_tokenizer("mobileclip_s2")

    # Encode all prompts
    tokens = tokenizer(prompts)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Save as binary (float32 array)
    embeddings = text_features.cpu().numpy()
    with open(OUTPUT_PATH, "wb") as f:
        for embedding in embeddings:
            f.write(struct.pack(f"{len(embedding)}f", *embedding))

    print(f"Saved {len(words)} embeddings ({embeddings.shape[1]}D) to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add MobileCLIP download and text embedding generation scripts"
```

---

## Task 17: Expand Vocabulary & Label Mappings

**Files:**
- Modify: `ARYA/Resources/vocabulary.json` (expand to full ~120 words)
- Modify: `ARYA/Resources/label_mappings.json` (expand with all VNClassify mappings)
- Create: `scripts/build_label_mappings.py`

- [ ] **Step 1: Create build_label_mappings.py**

This script helps generate the label mapping by querying VNClassify's known classifications:

```python
#!/usr/bin/env python3
"""Helper to generate label_mappings.json by listing VNClassify's taxonomy.

Run this on macOS to extract all known VNClassify labels, then manually map
each one to a child vocabulary word (or null to reject).

Usage: python3 build_label_mappings.py > vn_labels.txt
Then manually create label_mappings.json from the output.
"""

import subprocess
import json

# This script runs a Swift snippet to extract VNClassify labels
swift_code = '''
import Vision
let request = VNClassifyImageRequest()
let ids = try! request.supportedIdentifiers()
for id in ids.sorted() {
    print(id)
}
'''

print("Run the following in a Swift playground or command line to get all VNClassify labels:")
print("---")
print(swift_code)
print("---")
print("Then map each label to your vocabulary word or null in label_mappings.json")
```

- [ ] **Step 2: Expand vocabulary.json to full word list**

Update `ARYA/Resources/vocabulary.json` with all ~120 words from the spec (animals, food, home, kitchen, outdoors, clothing, bathroom, electronics, toys, school, body, other categories).

- [ ] **Step 3: Expand label_mappings.json**

Expand with comprehensive mappings for all VNClassify labels that map to vocabulary words. This requires running VNClassify's `supportedIdentifiers()` on a Mac and manually mapping each relevant label.

- [ ] **Step 4: Expand phoneme_to_letter_mappings.json**

Add hand-verified phoneme-to-letter mappings for all ~120 words using CMU Pronouncing Dictionary as reference.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: expand vocabulary to full ~120 words with label mappings"
```

---

## Task 18: Generate Audio & Timing Data

**Files:**
- Creates audio files and timing.json for each vocabulary word

- [ ] **Step 1: Set up Python environment**

```bash
cd scripts
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 2: Install Montreal Forced Aligner and download models**

```bash
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

- [ ] **Step 3: Generate audio files**

```bash
export ELEVENLABS_API_KEY="your-key-here"
python3 generate_audio.py
```

Expected: Audio files created in `ARYA/Resources/Vocabulary/<word>/audio.m4a` for each word

- [ ] **Step 4: Run forced alignment**

```bash
python3 align_audio.py
```

Expected: TextGrid files created alongside each audio file

- [ ] **Step 5: Build timing data**

```bash
python3 build_timing_data.py
```

Expected: `timing.json` files created in each word directory

- [ ] **Step 6: Verify timing data for a few words manually**

Spot-check `dog/timing.json`, `elephant/timing.json`, `chair/timing.json` — ensure phoneme boundaries are reasonable and letter indices are correct.

- [ ] **Step 7: Commit**

```bash
git add ARYA/Resources/Vocabulary/ && git commit -m "feat: add generated audio and timing data for full vocabulary"
```

---

## Task 19: Build MobileCLIP Embeddings

**Files:**
- Creates `ARYA/Resources/text_embeddings.bin`

- [ ] **Step 1: Download MobileCLIP model**

```bash
bash scripts/download_mobileclip.sh
```

- [ ] **Step 2: Generate text embeddings**

```bash
cd scripts && source venv/bin/activate
python3 build_clip_embeddings.py
```

Expected: `ARYA/Resources/text_embeddings.bin` created

- [ ] **Step 3: Compile CoreML model for Xcode**

The `.mlpackage` needs to be included in the Xcode project. Verify it's in the `ARYA/Resources/` directory and referenced in `project.yml`.

- [ ] **Step 4: Regenerate Xcode project and build**

```bash
cd /path/to/ARYA && xcodegen generate
xcodebuild build -project ARYA.xcodeproj -scheme ARYA -destination 'platform=iOS Simulator,name=iPhone 16'
```

Expected: BUILD SUCCEEDED

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add MobileCLIP CoreML model and pre-computed text embeddings"
```

---

## Task 20: Integration Testing on Device

**Files:** No new files — testing and tuning existing code

- [ ] **Step 1: Deploy to physical iPhone**

```bash
xcodebuild -project ARYA.xcodeproj -scheme ARYA -destination 'id=<DEVICE_UDID>' build
```

Or open `ARYA.xcodeproj` in Xcode and run on device.

- [ ] **Step 2: Test camera permission flow**

- Launch app → should prompt for camera permission
- Grant → should show camera feed
- Verify full-screen, no chrome, status bar hidden

- [ ] **Step 3: Test onboarding hint**

- First launch → should show pulsing hand-tap icon
- Tap a recognized object → hint should disappear permanently
- Kill and relaunch → hint should not appear

- [ ] **Step 4: Test object detection accuracy**

Test with common household objects:
- Point at a cup → tap → should recognize "cup"
- Point at a chair → tap → should recognize "chair"
- Point at a dog (photo or real) → tap → should recognize "dog"
- Point at something not in vocabulary → tap → should do nothing

Document accuracy results. If threshold tuning needed, adjust values in `ConsensusGate.swift`.

- [ ] **Step 5: Test speech and letter highlighting**

- Tap recognized object → verify audio plays
- Verify letters highlight in sync with pronunciation
- Verify "ph" in "elephant" highlights together
- Verify all letters glow after word completes
- Verify tapping anywhere returns to camera

- [ ] **Step 6: Test mask highlighting**

- Verify background dims around tapped object
- Verify object silhouette glows (not a bounding box)
- Verify glow follows actual object shape

- [ ] **Step 7: Tune confidence thresholds if needed**

Based on testing, adjust thresholds in `ConsensusGate.swift`:
- If too many false positives → raise thresholds
- If too many objects rejected → lower thresholds slightly
- Priority: accuracy over coverage

- [ ] **Step 8: Commit any tuning changes**

```bash
git add -A && git commit -m "fix: tune confidence thresholds based on device testing"
```

---

## Task 21: Polish & Final Verification

- [ ] **Step 1: Verify haptic feedback**

Ensure `UIImpactFeedbackGenerator(.light)` fires on tap.

- [ ] **Step 2: Verify animation timing**

- Dim/glow transition: 0.3s ease-in-out
- Letter transitions: 0.15s ease-in-out
- All-glow hold after word: 0.5s

- [ ] **Step 3: Test edge cases**

- Tap while already in learning mode → should dismiss and return to exploring
- Tap empty space (no object) → nothing should happen
- Move camera while in learning mode → overlay should stay stable
- Very dark environment → should gracefully do nothing (no phantom detections)
- Multiple objects close together → should only highlight the tapped one

- [ ] **Step 4: Verify app size**

Check the built IPA/app size. Target: ~40MB.

- [ ] **Step 5: Final commit**

```bash
git add -A && git commit -m "chore: polish and verify final app behavior"
```
