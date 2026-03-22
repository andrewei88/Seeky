# Fix All Core ARYA Issues — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 8 critical/secondary issues preventing ARYA from working: instance tracking drops, wrong speech engine, broken classification thresholds, white mask artifact, and tap tolerance.

**Architecture:** Each fix is isolated to 1-2 files with no cross-task dependencies beyond Task 3 (SegmentationResult change) which Tasks 4 and 5 depend on. Tasks are ordered so each produces a buildable state.

**Tech Stack:** Swift, AVFoundation (AVAudioPlayer), Vision framework, CoreImage, SwiftUI

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `ARYA/Speech/WordSpeaker.swift` | **Rewrite** | Replace AVSpeechSynthesizer with AVAudioPlayer for pre-recorded .m4a files |
| `ARYA/Views/LearningOverlayView.swift` | **Modify** | Remove timing scaling hack, use real audio duration |
| `ARYA/Detection/InstanceTracker.swift` | **Modify** | Add grace period for disappeared instances, add padded hit testing |
| `ARYA/Detection/ClassificationEngine.swift` | **Modify** | Fix rank calculation to skip null-mapped labels, lower thresholds |
| `ARYA/Detection/SegmentationEngine.swift` | **Modify** | Store VNImageRequestHandler in SegmentationResult for reuse |
| `ARYA/Views/MaskOverlayView.swift` | **Modify** | Use stored handler, add failure fallback |
| `ARYA/App/AppState.swift` | **Modify** | Freeze segmentation during learning mode, manage audio session |

---

### Task 1: Replace AVSpeechSynthesizer with AVAudioPlayer in WordSpeaker

**Why:** The app uses Apple's robotic TTS instead of the pre-recorded ElevenLabs .m4a files that already exist in the bundle at `Vocabulary/{word}/audio.m4a`. This is the #1 user-facing issue (unnatural, too-fast voice).

**Files:**
- Rewrite: `ARYA/Speech/WordSpeaker.swift`
- Modify: `ARYA/Views/LearningOverlayView.swift`

- [ ] **Step 1: Rewrite WordSpeaker to use AVAudioPlayer**

Replace the entire `WordSpeaker.swift` with:

```swift
import AVFoundation
import Combine

final class WordSpeaker: NSObject, ObservableObject {
    @Published private(set) var currentTime: Double = 0
    @Published private(set) var isPlaying: Bool = false
    @Published private(set) var totalDuration: Double = 0

    private var audioPlayer: AVAudioPlayer?
    private var displayLink: CADisplayLink?
    private var onComplete: (() -> Void)?

    func speak(word: String, from bundle: Bundle = .main, onComplete: @escaping () -> Void) {
        stop()
        self.onComplete = onComplete

        // Configure audio session — use .ambient to avoid interrupting camera capture session
        try? AVAudioSession.sharedInstance().setCategory(.ambient, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)

        // Load pre-recorded ElevenLabs audio from bundle
        guard let url = bundle.url(forResource: "audio", withExtension: "m4a", subdirectory: "Vocabulary/\(word)") else {
            print("[WordSpeaker] No audio file found for '\(word)', skipping")
            onComplete()
            return
        }

        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = self
            player.prepareToPlay()
            self.audioPlayer = player
            self.totalDuration = player.duration
            self.isPlaying = true
            startDisplayLink()
            player.play()
        } catch {
            print("[WordSpeaker] Failed to create player for '\(word)': \(error)")
            onComplete()
        }
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
        currentTime = totalDuration
        isPlaying = false
        stopDisplayLink()

        // Brief delay to show all-glow state before dismissing
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.onComplete?()
        }
    }
}
```

Key changes:
- `AVSpeechSynthesizer` → `AVAudioPlayer` loading `Vocabulary/{word}/audio.m4a`
- `estimatedDuration` → `totalDuration` (real duration from player)
- Audio session: `.ambient` instead of `.playback` to avoid camera session conflict
- `currentTime` reads directly from `audioPlayer.currentTime` (frame-accurate)
- Graceful fallback if audio file is missing

- [ ] **Step 2: Update LearningOverlayView to remove timing scaling hack**

In `ARYA/Views/LearningOverlayView.swift`, the scaling logic compensates for AVSpeechSynthesizer's unpredictable duration vs. the timing data. Since we now play the exact audio the timing was generated from, no scaling is needed.

Replace the body's timing section. The `estimatedDuration` property reference changes to `totalDuration`:

```swift
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
            // Mask-based highlighting
            if let segResult = segmentation {
                MaskOverlayView(
                    observation: segResult.observation,
                    instanceIndex: instanceIndex,
                    pixelBuffer: segResult.pixelBuffer,
                    requestHandler: segResult.requestHandler
                )
                .ignoresSafeArea()
            } else {
                // Fallback dim overlay when no segmentation data
                Color.black.opacity(0.6)
                    .ignoresSafeArea()
            }

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

Changes:
- Removed the `timingDuration / speechDuration` scaling math entirely
- `wordSpeaker.currentTime` is used directly (it now comes from AVAudioPlayer, matching the timing data)
- Added `requestHandler` parameter to `MaskOverlayView` (for Task 4)

- [ ] **Step 3: Verify build compiles**

Run: `cd /Users/andrewwei/Projects/ARYA && xcodebuild -scheme ARYA -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED (may have warnings, no errors)

- [ ] **Step 4: Commit**

```bash
git add ARYA/Speech/WordSpeaker.swift ARYA/Views/LearningOverlayView.swift
git commit -m "fix: replace AVSpeechSynthesizer with pre-recorded ElevenLabs audio

WordSpeaker now plays Vocabulary/{word}/audio.m4a via AVAudioPlayer instead
of generating robotic TTS. Timing scaling removed since audio matches timing data.
Audio session changed to .ambient to avoid camera capture conflicts."
```

---

### Task 2: Add grace period and tap padding to InstanceTracker

**Why:** The tracker drops all instances when segmentation returns 0 for a single frame (common during camera motion). Also, exact bounding box hit testing is too strict for finger taps.

**Files:**
- Modify: `ARYA/Detection/InstanceTracker.swift`

- [ ] **Step 1: Add grace period and padded hit testing**

Replace `ARYA/Detection/InstanceTracker.swift` with:

```swift
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
    let gracePeriod: Double
    let tapPadding: CGFloat
    private let iouThreshold: Double
    private(set) var trackedInstances: [TrackedInstance] = []

    init(stabilityDuration: Double = 0.3, iouThreshold: Double = 0.3, gracePeriod: Double = 1.5, tapPadding: CGFloat = 0.04) {
        self.stabilityDuration = stabilityDuration
        self.iouThreshold = iouThreshold
        self.gracePeriod = gracePeriod
        self.tapPadding = tapPadding
    }

    mutating func update(instances: [DetectedInstance], timestamp: Double) {
        // Match new instances to existing tracked instances via IoU
        var matched = Set<Int>() // indices into trackedInstances that were matched
        var newTracked: [TrackedInstance] = []

        for instance in instances {
            if let matchIndex = bestMatch(for: instance) {
                // Existing instance — update position and timestamp
                var tracked = trackedInstances[matchIndex]
                tracked.currentInstance = instance
                tracked.lastSeenTimestamp = timestamp
                newTracked.append(tracked)
                matched.insert(matchIndex)
            } else {
                // New instance
                newTracked.append(TrackedInstance(
                    currentInstance: instance,
                    firstSeenTimestamp: timestamp,
                    lastSeenTimestamp: timestamp
                ))
            }
        }

        // Keep unmatched instances alive during grace period
        for (index, tracked) in trackedInstances.enumerated() {
            if !matched.contains(index) && (timestamp - tracked.lastSeenTimestamp) < gracePeriod {
                newTracked.append(tracked)
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
        // Use padded bounding boxes for more forgiving tap detection
        trackedInstances
            .map(\.currentInstance)
            .first { paddedBox($0.boundingBox).contains(point) }
    }

    private func paddedBox(_ rect: CGRect) -> CGRect {
        CGRect(
            x: max(0, rect.origin.x - tapPadding),
            y: max(0, rect.origin.y - tapPadding),
            width: min(1.0 - max(0, rect.origin.x - tapPadding), rect.width + tapPadding * 2),
            height: min(1.0 - max(0, rect.origin.y - tapPadding), rect.height + tapPadding * 2)
        )
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

Key changes:
- **Grace period (1.5s):** Unmatched instances survive for 1.5s after last seen instead of being immediately dropped
- **Tap padding (0.04 normalized):** Bounding boxes expanded ~4% in each direction for hit testing
- **Stability reduced (0.5→0.3s):** Objects become tappable faster
- Matched set tracking prevents duplicating instances that are both matched and in grace period

- [ ] **Step 2: Verify build compiles**

Run: `cd /Users/andrewwei/Projects/ARYA && xcodebuild -scheme ARYA -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add ARYA/Detection/InstanceTracker.swift
git commit -m "fix: add grace period and tap padding to InstanceTracker

Tracked instances now survive 1.5s after disappearing from segmentation
instead of being dropped on a single empty frame. Hit testing uses padded
bounding boxes for more forgiving finger tap detection."
```

---

### Task 3: Fix VN-only classification thresholds

**Why:** Without CLIP (model files don't exist), the VN-only path rejects correct classifications because it counts null-mapped generic categories in the rank. A laptop correctly identified but at rank 4 (after document, screenshot, machine, consumer_electronics — all null-mapped) gets rejected.

**Files:**
- Modify: `ARYA/Detection/ClassificationEngine.swift`

- [ ] **Step 1: Fix rank calculation and thresholds**

Replace `ARYA/Detection/ClassificationEngine.swift` with:

```swift
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
    }

    /// Classify a cropped object image. Returns the child-friendly word or nil.
    func classify(imageBuffer: CVPixelBuffer) async -> ClassificationResult? {
        async let vnResult = runVNClassify(imageBuffer)
        async let clipResult = runCLIP(imageBuffer)

        let observations = await vnResult
        let clip = await clipResult

        guard let observations = observations, !observations.isEmpty else {
            print("[Classify] VNClassify returned nil")
            return nil
        }

        // Log top 10 for debugging
        let top10 = observations.prefix(10).map { "\($0.identifier)(\(String(format: "%.3f", $0.confidence)))" }
        print("[VNClassify] Top 10: \(top10.joined(separator: ", "))")

        // Scan through ALL observations to find mapped words, skipping null-mapped labels
        // Track the best mapped word and a runner-up for margin comparison
        var bestMatch: (word: String, confidence: Float, rawIndex: Int)?
        var secondBestMatch: (word: String, confidence: Float)?

        for (index, obs) in observations.enumerated() {
            // Stop scanning at very low confidence
            if obs.confidence < 0.02 { break }

            let label = obs.identifier
            // Try all label variants
            let mapped = labelMapper.childWord(for: label)
                ?? labelMapper.childWord(for: label.replacingOccurrences(of: "_", with: " "))
                ?? labelMapper.childWord(for: label.lowercased().replacingOccurrences(of: "_", with: " "))

            // nil means not in whitelist — skip (unknown label)
            // labelMapper returns Optional<String?>: outer nil = not mapped, inner nil = explicitly rejected
            // If childWord returns nil, it's either unmapped or explicitly rejected — either way, skip
            guard let word = mapped else { continue }

            if bestMatch == nil {
                bestMatch = (word: word, confidence: obs.confidence, rawIndex: index)
            } else if word != bestMatch!.word && secondBestMatch == nil {
                secondBestMatch = (word: word, confidence: obs.confidence)
            }

            // Once we have best and second-best, we can decide
            if bestMatch != nil && secondBestMatch != nil { break }
        }

        guard let best = bestMatch else {
            print("[Classify] No mapped word found in results")
            return nil
        }

        print("[Classify] Best mapped: '\(best.word)' (conf=\(best.confidence), rawIndex=\(best.rawIndex)), " +
              "second: '\(secondBestMatch?.word ?? "none")' (conf=\(secondBestMatch?.confidence ?? 0))")

        // If no CLIP available, use VN alone with permissive thresholds
        guard let clip = clip else {
            // The word passed our whitelist — the main risk is confusion between two mapped words.
            // If there's a clear margin over the second mapped word, accept it.
            let margin = secondBestMatch != nil ? best.confidence / secondBestMatch!.confidence : Float.infinity
            let minConfidence: Float = 0.03 // Very low floor — whitelist is the real filter

            guard best.confidence >= minConfidence else {
                print("[Classify] VN-only: rejected '\(best.word)' (conf=\(best.confidence) < \(minConfidence))")
                return nil
            }

            // If two different mapped words are close in confidence, reject (ambiguous)
            if let second = secondBestMatch, margin < 1.5 {
                print("[Classify] VN-only: rejected '\(best.word)' — ambiguous with '\(second.word)' (margin=\(String(format: "%.2f", margin)))")
                return nil
            }

            print("[Classify] VN-only: accepted '\(best.word)' (conf=\(best.confidence), margin=\(String(format: "%.2f", margin)))")
            return ClassificationResult(word: best.word)
        }

        // Full dual-model consensus
        let secondConfidence = secondBestMatch.map { Double($0.confidence) } ?? 0

        guard let word = consensusGate.evaluate(
            vnClassifyWord: best.word,
            vnConfidence: Double(best.confidence),
            vnSecondConfidence: secondConfidence,
            clipWord: clip.top1.word,
            clipSimilarity: clip.top1.similarity,
            clipSecondSimilarity: clip.top2.similarity
        ) else {
            return nil
        }

        return ClassificationResult(word: word)
    }

    private func runVNClassify(_ buffer: CVPixelBuffer) async -> [VNClassificationObservation]? {
        await withCheckedContinuation { continuation in
            let request = VNClassifyImageRequest { request, error in
                guard let observations = request.results as? [VNClassificationObservation],
                      !observations.isEmpty else {
                    continuation.resume(returning: nil)
                    return
                }

                continuation.resume(returning: observations)
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

Key changes:
- **Rank now skips null-mapped and unmapped labels entirely.** `bestMatch` is the first observation that maps to a real child word. The laptop at raw index 4 becomes the best match because indices 0-3 are all null-mapped.
- **Minimum confidence lowered to 0.03.** The whitelist IS the filter — if VNClassify says "computer" and our whitelist maps that to "laptop", we trust it unless confidence is absurdly low.
- **Ambiguity check:** Instead of raw rank thresholds, we compare the best mapped word against the second-best mapped word. If they're close (margin < 1.5x), we reject as ambiguous.
- **Scan floor lowered to 0.02** (was 0.05) to find more mapped labels.

- [ ] **Step 2: Verify build compiles**

Run: `cd /Users/andrewwei/Projects/ARYA && xcodebuild -scheme ARYA -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add ARYA/Detection/ClassificationEngine.swift
git commit -m "fix: classification now skips null-mapped labels when calculating rank

VNClassify ranks generic categories (document, screenshot, machine) above
real objects. These are null-mapped in our whitelist but were counted in rank,
causing valid matches like laptop to be rejected. Now only compares confidence
between actual mapped child words."
```

---

### Task 4: Fix SegmentationResult to carry VNImageRequestHandler

**Why:** MaskOverlayView creates a NEW VNImageRequestHandler to generate masks, which may produce different/corrupt results. The original handler used during segmentation must be reused.

**Files:**
- Modify: `ARYA/Detection/SegmentationEngine.swift`
- Modify: `ARYA/Views/MaskOverlayView.swift`

- [ ] **Step 1: Add requestHandler to SegmentationResult**

In `ARYA/Detection/SegmentationEngine.swift`, modify `SegmentationResult` to store the handler, and return it from `segment()`:

```swift
import Vision
import CoreImage
import UIKit

struct SegmentationResult {
    let instances: [DetectedInstance]
    let observation: VNInstanceMaskObservation?
    let pixelBuffer: CVPixelBuffer
    let requestHandler: VNImageRequestHandler?
}

final class SegmentationEngine {
    func segment(pixelBuffer: CVPixelBuffer) -> SegmentationResult {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        do {
            try handler.perform([request])
        } catch {
            print("[Segmentation] Error: \(error.localizedDescription)")
            return SegmentationResult(instances: [], observation: nil, pixelBuffer: pixelBuffer, requestHandler: nil)
        }

        guard let observation = request.results?.first else {
            return SegmentationResult(instances: [], observation: nil, pixelBuffer: pixelBuffer, requestHandler: nil)
        }

        let allInstances = observation.allInstances
        var detected: [DetectedInstance] = []

        for index in allInstances {
            if let mask = try? observation.generateScaledMaskForImage(forInstances: IndexSet(integer: index), from: handler) {
                let boundingBox = computeBoundingBox(from: mask)
                guard boundingBox.width > 0.01 && boundingBox.height > 0.01 else { continue }
                detected.append(DetectedInstance(id: index, boundingBox: boundingBox))
            }
        }

        return SegmentationResult(instances: detected, observation: observation, pixelBuffer: pixelBuffer, requestHandler: handler)
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

        return CGRect(
            x: CGFloat(minX) / CGFloat(width),
            y: CGFloat(minY) / CGFloat(height),
            width: CGFloat(maxX - minX) / CGFloat(width),
            height: CGFloat(maxY - minY) / CGFloat(height)
        )
    }
}
```

- [ ] **Step 2: Update MaskOverlayView to use stored handler with fallback**

Replace `ARYA/Views/MaskOverlayView.swift`:

```swift
import SwiftUI
import Vision
import CoreImage

struct MaskOverlayView: UIViewRepresentable {
    let observation: VNInstanceMaskObservation?
    let instanceIndex: Int
    let pixelBuffer: CVPixelBuffer?
    let requestHandler: VNImageRequestHandler?

    func makeUIView(context: Context) -> MaskUIView {
        MaskUIView()
    }

    func updateUIView(_ uiView: MaskUIView, context: Context) {
        uiView.updateMask(observation: observation, instanceIndex: instanceIndex, pixelBuffer: pixelBuffer, requestHandler: requestHandler)
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

    func updateMask(observation: VNInstanceMaskObservation?, instanceIndex: Int, pixelBuffer: CVPixelBuffer?, requestHandler: VNImageRequestHandler?) {
        guard let observation = observation, let requestHandler = requestHandler else {
            showFallbackDim()
            return
        }

        guard let maskBuffer = try? observation.generateScaledMaskForImage(
            forInstances: IndexSet(integer: instanceIndex),
            from: requestHandler
        ) else {
            showFallbackDim()
            return
        }

        let maskCI = CIImage(cvPixelBuffer: maskBuffer)

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
        } else {
            showFallbackDim()
            return
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

    private func showFallbackDim() {
        // Show a simple semi-transparent overlay instead of a white artifact
        dimLayer.contents = nil
        dimLayer.backgroundColor = UIColor.black.withAlphaComponent(0.6).cgColor
        glowLayer.contents = nil
    }
}
```

Key changes:
- `requestHandler` parameter added — reuses the original handler from segmentation
- `showFallbackDim()` — on any mask failure, shows a clean dark overlay instead of white artifact
- No more creating a new `VNImageRequestHandler` inside the view

- [ ] **Step 3: Verify build compiles**

Run: `cd /Users/andrewwei/Projects/ARYA && xcodebuild -scheme ARYA -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add ARYA/Detection/SegmentationEngine.swift ARYA/Views/MaskOverlayView.swift
git commit -m "fix: reuse original VNImageRequestHandler for mask generation

SegmentationResult now carries the VNImageRequestHandler used during
segmentation. MaskOverlayView reuses it instead of creating a new one,
eliminating white artifacts from mismatched handlers. Added fallback dim
overlay when mask generation fails."
```

---

### Task 5: Freeze segmentation during learning mode

**Why:** During learning mode, the camera keeps running and overwriting `latestSegmentation`. This causes the mask overlay to re-render with new frames where the instance index may no longer be valid, producing visual glitches and the white artifact.

**Files:**
- Modify: `ARYA/App/AppState.swift`

- [ ] **Step 1: Add frozen segmentation and freeze/unfreeze logic**

In `ARYA/App/AppState.swift`:

```swift
import SwiftUI
import Vision

enum AppMode: Equatable {
    case exploring
    case classifying
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

    // Live segmentation (updates every frame)
    private var liveSegmentation: SegmentationResult?

    // Frozen segmentation for learning mode (doesn't update)
    @Published var frozenSegmentation: SegmentationResult?

    // Public accessor: frozen during learning, live during exploring
    var currentSegmentation: SegmentationResult? {
        if case .learning = mode {
            return frozenSegmentation
        }
        return liveSegmentation
    }

    init() {
        hasCompletedFirstTap = UserDefaults.standard.bool(forKey: "hasCompletedFirstTap")
        vocabularyStore = VocabularyStore.load()
        labelMapper = LabelMapper.load()

        let clipEmbeddings = CLIPEmbeddings.load(vocabulary: vocabularyStore.entries.map(\.word))
        classificationEngine = ClassificationEngine(labelMapper: labelMapper, clipEmbeddings: clipEmbeddings)

        cameraManager.delegate = self
    }

    func handleTap(at normalizedPoint: CGPoint) {
        guard mode == .exploring else { return }

        // Transform tap from portrait screen space to landscape image space.
        // Camera delivers landscape buffers (1920x1080) rotated 90° CW for portrait display.
        // Portrait (screenX, screenY) → Landscape image (screenY, 1 - screenX)
        let imagePoint = CGPoint(x: normalizedPoint.y, y: 1.0 - normalizedPoint.x)

        print("[Tap] screen=\(normalizedPoint) → image=\(imagePoint), tracked=\(instanceTracker.trackedInstances.count) instances")
        for tracked in instanceTracker.trackedInstances {
            print("  instance id=\(tracked.currentInstance.id) bbox=\(tracked.currentInstance.boundingBox)")
        }

        // Check if tap hit a tracked instance
        guard let instance = instanceTracker.instance(at: imagePoint) else {
            print("[Tap] No instance hit")
            return
        }

        // Trigger haptic
        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()

        mode = .classifying

        guard let segResult = liveSegmentation else {
            mode = .exploring
            return
        }

        print("[Tap] Hit instance id=\(instance.id), classifying...")

        Task {
            // Crop the instance region from the full frame
            let croppedBuffer = cropPixelBuffer(segResult.pixelBuffer, to: instance.boundingBox)

            guard let croppedBuffer = croppedBuffer else {
                print("[Tap] Failed to crop pixel buffer")
                mode = .exploring
                return
            }

            guard let result = await classificationEngine.classify(imageBuffer: croppedBuffer) else {
                print("[Tap] Classification returned nil (thresholds not met)")
                mode = .exploring
                return
            }

            // Mark first tap complete
            if !hasCompletedFirstTap {
                hasCompletedFirstTap = true
                UserDefaults.standard.set(true, forKey: "hasCompletedFirstTap")
            }

            // Freeze segmentation before entering learning mode
            frozenSegmentation = segResult
            mode = .learning(word: result.word, instanceIndex: instance.id)
        }
    }

    func dismissLearning() {
        wordSpeaker.stop()
        frozenSegmentation = nil
        mode = .exploring
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

extension AppState: CameraManagerDelegate {
    nonisolated func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        let result = segmentationEngine.segment(pixelBuffer: pixelBuffer)
        let timeSeconds = CMTimeGetSeconds(timestamp)

        if !result.instances.isEmpty {
            print("[Seg] Found \(result.instances.count) instances at t=\(String(format: "%.1f", timeSeconds))")
        }

        Task { @MainActor in
            instanceTracker.update(instances: result.instances, timestamp: timeSeconds)
            liveSegmentation = result
        }
    }
}
```

Key changes:
- `latestSegmentation` split into `liveSegmentation` (always updating) and `frozenSegmentation` (locked at tap time)
- `frozenSegmentation` is set just before entering `.learning` mode, ensuring the mask/observation/handler combo is consistent
- `dismissLearning()` clears the frozen data
- `currentSegmentation` computed property returns the right one based on mode

- [ ] **Step 2: Update ContentView to use frozenSegmentation**

In `ARYA/Views/ContentView.swift`, change the segmentation parameter passed to LearningOverlayView:

Find: `segmentation: appState.latestSegmentation,`
Replace with: `segmentation: appState.frozenSegmentation,`

Full updated ContentView:

```swift
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
                    segmentation: appState.frozenSegmentation,
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

- [ ] **Step 3: Verify build compiles**

Run: `cd /Users/andrewwei/Projects/ARYA && xcodebuild -scheme ARYA -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add ARYA/App/AppState.swift ARYA/Views/ContentView.swift
git commit -m "fix: freeze segmentation data during learning mode

Segmentation now splits into live (always updating) and frozen (locked at
tap time). Learning overlay uses frozen data to prevent mask flickering and
white artifacts from stale observation/buffer mismatches."
```

---

### Task 6: Final integration verification

- [ ] **Step 1: Full clean build**

Run: `cd /Users/andrewwei/Projects/ARYA && xcodebuild clean build -scheme ARYA -destination 'platform=iOS Simulator,name=iPhone 16' 2>&1 | tail -10`
Expected: BUILD SUCCEEDED with no errors

- [ ] **Step 2: Verify all audio resources accessible**

Quick sanity check that audio files are reachable from bundle:

Run: `ls /Users/andrewwei/Projects/ARYA/ARYA/Resources/Vocabulary/laptop/audio.m4a /Users/andrewwei/Projects/ARYA/ARYA/Resources/Vocabulary/dog/audio.m4a /Users/andrewwei/Projects/ARYA/ARYA/Resources/Vocabulary/cat/audio.m4a`
Expected: All three files listed

- [ ] **Step 3: Commit any remaining changes and tag**

```bash
git add -A
git status
# Only commit if there are changes
```
