import CoreMedia
import os
import SwiftUI

enum AppMode: Equatable {
    case exploring
    case classifying
    case learning(word: String)
    case quizPrompting      // showing target word, waiting for child to tap
    case quizClassifying    // child tapped, running classifier
    case quizResult         // showing correct/wrong feedback
}

enum QuizAnswerResult: Equatable {
    case correct
    case wrong(actual: String?)  // what the model identified (nil if rejected)
}

/// A single challenge in a quiz session.
enum ChallengeTarget: Equatable {
    case word(String)               // "Find the cup"
    case category(String)           // "Find an animal"
}

struct Challenge: Equatable {
    let target: ChallengeTarget

    /// The display text for the prompt (e.g. "cup" or "animal").
    var displayText: String {
        switch target {
        case .word(let w): return w
        case .category(let c): return c
        }
    }

    /// Check if a classified word satisfies this challenge.
    func matches(word: String) -> Bool {
        switch target {
        case .word(let target): return word == target
        case .category(let category):
            guard let words = WordProgressStore.quizCategories[category] else { return false }
            return words.contains(word)
        }
    }
}

/// Manages a quiz session: challenge list, current index, attempts, results.
@MainActor
final class QuizSession: ObservableObject {
    let challenges: [Challenge]
    @Published var currentIndex: Int = 0
    @Published var attemptsOnCurrent: Int = 0
    @Published var lastResult: QuizAnswerResult?
    @Published var results: [(challenge: Challenge, correct: Bool, foundWord: String?)] = []

    /// The word found on correct category challenges (shown briefly after match).
    @Published var lastFoundWord: String?

    static let maxAttempts = 3
    static let challengesPerSession = 5

    /// Convenience: all display words for dot tracking (backwards compat).
    var words: [String] { challenges.map(\.displayText) }

    var currentChallenge: Challenge? {
        guard currentIndex < challenges.count else { return nil }
        return challenges[currentIndex]
    }

    /// The text to display for the current challenge.
    var currentWord: String? { currentChallenge?.displayText }

    var isComplete: Bool { currentIndex >= challenges.count }
    var correctCount: Int { results.filter(\.correct).count }
    var canGoBack: Bool { currentIndex > 0 }

    init(challenges: [Challenge]) {
        self.challenges = challenges
    }

    /// Convenience init for word-only sessions (backwards compat).
    convenience init(words: [String]) {
        self.init(challenges: words.map { Challenge(target: .word($0)) })
    }

    func recordResult(correct: Bool, foundWord: String? = nil) {
        guard let challenge = currentChallenge else { return }
        results.append((challenge: challenge, correct: correct, foundWord: foundWord))
        lastFoundWord = correct ? foundWord : nil
    }

    func advance() {
        attemptsOnCurrent = 0
        lastResult = nil
        lastFoundWord = nil
        currentIndex += 1
    }

    func goBack() {
        guard canGoBack else { return }
        if !results.isEmpty && results.count >= currentIndex {
            results.removeLast()
        }
        currentIndex -= 1
        attemptsOnCurrent = 0
        lastResult = nil
        lastFoundWord = nil
    }
}

@MainActor
final class AppState: ObservableObject {
    private static let hasCompletedFirstTapKey = "hasCompletedFirstTap"

    @Published var mode: AppMode = .exploring
    @Published private(set) var hasCompletedFirstTap: Bool

    let cameraManager = CameraManager()
    let segmentationEngine = SegmentationEngine()
    let wordSpeaker = WordSpeaker()
    let vocabularyStore: VocabularyStore
    let classificationEngine: ClassificationEngine
    let correctionStore = CorrectionStore()
    let trainingCapture = TrainingCapture()
    let wordProgressStore = WordProgressStore()
    @Published var quizSession: QuizSession?
    private var instanceTracker = InstanceTracker(tapPadding: 0.08)

    private var liveSegmentation: SegmentationResult?

    @Published var tapScreenPoint: CGPoint = .zero

    private(set) var lastFeatures: [Float]?
    private var lastCroppedBuffer: CVPixelBuffer?

    @Published var showingCorrectionPicker = false
    @Published var showingParentSettings = false

    /// Location for quiz word filtering. nil = use all locations.
    @Published var selectedLocation: WordLocation? = nil {
        didSet { UserDefaults.standard.set(selectedLocation?.rawValue, forKey: "selectedLocation") }
    }

    /// Parent-selected categories for focused hunts. Empty = all categories.
    @Published var selectedCategories: Set<String> = [] {
        didSet { UserDefaults.standard.set(Array(selectedCategories), forKey: "selectedCategories") }
    }

    /// Generation counter to invalidate stale delayed callbacks (retry/advance).
    /// Incremented on every quiz state transition (tap, correct, wrong, skip, advance).
    private var quizGeneration: Int = 0

    /// Idle timeout: skip segmentation after this many seconds with no taps.
    /// Uses atomic storage so the camera delegate (nonisolated) can read it safely.
    private let idleTimeout: TimeInterval = 30
    private let _lastTapTime = OSAllocatedUnfairLock(initialState: Date())

    init() {
        hasCompletedFirstTap = UserDefaults.standard.bool(forKey: Self.hasCompletedFirstTapKey)
        vocabularyStore = VocabularyStore.load()

        // Restore persisted location and category selections
        if let locRaw = UserDefaults.standard.string(forKey: "selectedLocation"),
           let loc = WordLocation(rawValue: locRaw) {
            selectedLocation = loc
        }
        if let cats = UserDefaults.standard.array(forKey: "selectedCategories") as? [String] {
            selectedCategories = Set(cats)
        }

        classificationEngine = ClassificationEngine(
            correctionStore: correctionStore
        )

        cameraManager.delegate = self

        // Load the ML model in the background so the camera starts immediately.
        // Classification returns nil until the model is ready.
        let loadStart = CFAbsoluteTimeGetCurrent()
        Task.detached(priority: .userInitiated) {
            let classifier = CustomClassifier()
            let elapsed = CFAbsoluteTimeGetCurrent() - loadStart
            await MainActor.run {
                self.classificationEngine.customClassifier = classifier
                print("[AppState] Custom classifier ready in \(String(format: "%.2f", elapsed))s")
            }
        }
    }

    var bufferIsLandscape: Bool = false
    private var hasLoggedBufferDims = false

    var showGlow: Bool {
        if case .learning = mode { return true }
        if showingCorrectionPicker { return true }
        return false
    }

    func handleTap(imagePoint: CGPoint, screenPoint: CGPoint) {
        guard mode == .exploring else { return }

        _lastTapTime.withLock { $0 = Date() }

        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()

        tapScreenPoint = screenPoint
        mode = .classifying

        guard let segResult = liveSegmentation else {
            print("[Tap] No segmentation data available")
            let errorGenerator = UINotificationFeedbackGenerator()
            errorGenerator.notificationOccurred(.error)
            mode = .exploring
            return
        }

        let bufW = CVPixelBufferGetWidth(segResult.pixelBuffer)
        let bufH = CVPixelBufferGetHeight(segResult.pixelBuffer)

        // captureDevicePointConverted returns coordinates in the capture device's
        // native sensor space (landscape). The pixel buffer is rotated 90° CW to portrait
        // via videoRotationAngle=90. Convert: buffer(x,y) = (1 - device.y, device.x)
        let bufferPoint = CGPoint(x: 1.0 - imagePoint.y, y: imagePoint.x)

        let cropRect = tapCenteredCropRect(tapPoint: bufferPoint, cropFraction: 0.25,
                                           bufferWidth: bufW, bufferHeight: bufH)
        print("[Tap] screen=\(screenPoint) → device=\(imagePoint) → buffer=\(bufferPoint), crop=\(cropRect)")

        Task {
            let croppedBuffer = cropPixelBuffer(segResult.pixelBuffer, to: cropRect)

            guard let croppedBuffer = croppedBuffer else {
                print("[Tap] Failed to crop pixel buffer")
                mode = .exploring
                return
            }

            guard let result = await classificationEngine.classify(imageBuffer: croppedBuffer) else {
                print("[Tap] Classification returned nil (no features)")
                let errorGenerator = UINotificationFeedbackGenerator()
                errorGenerator.notificationOccurred(.error)
                mode = .exploring
                return
            }

            if !hasCompletedFirstTap {
                hasCompletedFirstTap = true
                UserDefaults.standard.set(true, forKey: Self.hasCompletedFirstTapKey)
            }

            lastFeatures = result.features
            lastCroppedBuffer = croppedBuffer

            if let word = result.word {
                wordProgressStore.recordExploreIdentification(word: word)
                mode = .learning(word: word)
            } else {
                // Consensus gate rejected — return to exploring with haptic feedback.
                // Parent can use the pencil button during learning mode to correct.
                print("[Tap] Unrecognized object — returning to exploring")
                let errorGenerator = UINotificationFeedbackGenerator()
                errorGenerator.notificationOccurred(.warning)
                mode = .exploring
            }
        }
    }

    func dismissLearning() {
        wordSpeaker.stop()
        showingCorrectionPicker = false
        mode = .exploring
    }

    func startCorrection() {
        wordSpeaker.stop()
        showingCorrectionPicker = true
    }

    func undoLastCorrection() {
        if let word = correctionStore.undoLastCorrection() {
            print("[AppState] Undid correction for '\(word)'")
        }
        showingCorrectionPicker = false
        mode = .exploring
    }

    func applyCorrection(word: String) {
        guard let embedding = lastFeatures else {
            print("[Correction] No feature embedding available for correction")
            showingCorrectionPicker = false
            return
        }
        correctionStore.addCorrection(embedding: embedding, word: word)
        wordProgressStore.recordExploreCorrection(word: word)
        if let buffer = lastCroppedBuffer {
            trainingCapture.save(imageBuffer: buffer, word: word)
        }
        showingCorrectionPicker = false
        mode = .learning(word: word)
    }

    // MARK: - Quiz Mode

    func handleQuizTap(imagePoint: CGPoint, screenPoint: CGPoint) {
        guard mode == .quizPrompting, let session = quizSession, let challenge = session.currentChallenge else { return }

        _lastTapTime.withLock { $0 = Date() }
        quizGeneration += 1
        let gen = quizGeneration

        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()

        tapScreenPoint = screenPoint
        mode = .quizClassifying

        guard let segResult = liveSegmentation else {
            print("[Quiz] No segmentation data")
            mode = .quizPrompting
            return
        }

        let bufW = CVPixelBufferGetWidth(segResult.pixelBuffer)
        let bufH = CVPixelBufferGetHeight(segResult.pixelBuffer)
        let bufferPoint = CGPoint(x: 1.0 - imagePoint.y, y: imagePoint.x)
        let cropRect = tapCenteredCropRect(tapPoint: bufferPoint, cropFraction: 0.25,
                                           bufferWidth: bufW, bufferHeight: bufH)

        Task {
            let croppedBuffer = cropPixelBuffer(segResult.pixelBuffer, to: cropRect)
            guard let croppedBuffer = croppedBuffer else {
                print("[Quiz] Failed to crop")
                mode = .quizPrompting
                return
            }

            guard let result = await classificationEngine.classify(imageBuffer: croppedBuffer) else {
                print("[Quiz] Classification returned nil")
                session.attemptsOnCurrent += 1
                session.lastResult = .wrong(actual: nil)
                mode = .quizResult
                return
            }

            lastFeatures = result.features
            lastCroppedBuffer = croppedBuffer

            if let word = result.word, challenge.matches(word: word) {
                // Correct! Celebrate then auto-advance
                session.lastResult = .correct
                session.lastFoundWord = word
                let successGenerator = UINotificationFeedbackGenerator()
                successGenerator.notificationOccurred(.success)
                let foundWord = word
                print("[Quiz] Correct! Found '\(foundWord)' for challenge '\(challenge.displayText)'")

                wordSpeaker.speakCelebration(word: foundWord) { [weak self] in
                    // Auto-advance after celebration audio finishes
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                        guard let self, self.quizGeneration == gen else { return }
                        self.advanceQuiz()
                    }
                }
            } else {
                // Wrong — shake + haptic only. No audio naming (risk of teaching wrong info).
                session.attemptsOnCurrent += 1
                session.lastResult = .wrong(actual: result.word)
                let errorGenerator = UINotificationFeedbackGenerator()
                errorGenerator.notificationOccurred(.error)
                print("[Quiz] Wrong — expected '\(challenge.displayText)', got '\(result.word ?? "nil")'")

                let hasRetries = session.attemptsOnCurrent < QuizSession.maxAttempts

                // Brief pause for shake animation, then auto-retry or advance.
                // Uses generation counter to prevent stale callbacks from interfering
                // with later correct answers (the root cause of the quiz freeze bug).
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { [weak self] in
                    guard let self, self.quizGeneration == gen else { return }
                    if hasRetries {
                        self.retryQuizWord()
                    } else {
                        self.advanceQuiz()
                    }
                }
            }
            mode = .quizResult
        }
    }

    /// Parent overrides the quiz result (pencil button). Flips correct to wrong or wrong to correct.
    func overrideQuizResult() {
        guard let session = quizSession, let challenge = session.currentChallenge else { return }
        let displayText = challenge.displayText

        if case .correct = session.lastResult {
            // Parent says this was actually wrong
            session.lastResult = .wrong(actual: displayText)
            print("[Quiz] Parent overrode to WRONG for '\(displayText)'")
        } else {
            // Parent says the child was actually right (model error)
            session.lastResult = .correct
            print("[Quiz] Parent overrode to CORRECT for '\(displayText)'")

            // Also save the correction for future model improvement
            if case .word(let targetWord) = challenge.target {
                if let embedding = lastFeatures {
                    correctionStore.addCorrection(embedding: embedding, word: targetWord)
                }
                if let buffer = lastCroppedBuffer {
                    trainingCapture.save(imageBuffer: buffer, word: targetWord)
                }
            }
        }

        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
    }

    /// Advance to the next quiz word (called after result is shown).
    func advanceQuiz() {
        guard let session = quizSession, let challenge = session.currentChallenge else { return }
        wordSpeaker.stop()

        // Record the result
        if let result = session.lastResult {
            let correct = result == .correct
            let foundWord = session.lastFoundWord
            session.recordResult(correct: correct, foundWord: foundWord)

            // Track progress for the found word (or the target word for word challenges)
            switch challenge.target {
            case .word(let targetWord):
                if correct {
                    wordProgressStore.recordQuizCorrect(word: targetWord)
                } else {
                    wordProgressStore.recordQuizWrong(word: targetWord)
                }
            case .category:
                // For category challenges, track the word they actually found
                if correct, let found = foundWord {
                    wordProgressStore.recordQuizCorrect(word: found)
                }
            }
        }

        session.advance()

        if session.isComplete {
            print("[Quiz] Session complete: \(session.correctCount)/\(session.words.count)")
            // Stay in quizResult mode — UI will show completion
        } else {
            mode = .quizPrompting
            speakCurrentQuizWord()
        }
    }

    /// Skip the current quiz word (target not in room).
    func skipQuizWord() {
        guard let session = quizSession else { return }
        wordSpeaker.stop()
        // Don't record anything — skip doesn't count as right or wrong
        session.advance()

        if session.isComplete {
            print("[Quiz] Session complete: \(session.correctCount)/\(session.words.count)")
        } else {
            mode = .quizPrompting
            speakCurrentQuizWord()
        }
    }

    /// Try again on the same quiz word (after wrong answer, if attempts remain).
    /// Does not re-speak the prompt — the child already knows what to look for.
    func retryQuizWord() {
        mode = .quizPrompting
    }

    /// Replay the current quiz word's audio prompt.
    func replayQuizWord() {
        speakCurrentQuizWord()
    }

    /// Go back to the previous quiz word.
    func goBackQuizWord() {
        guard let session = quizSession, session.canGoBack else { return }
        wordSpeaker.stop()
        session.goBack()
        mode = .quizPrompting
        speakCurrentQuizWord()
    }

    /// Start a new scavenger hunt session (called on app launch and when returning from explore).
    func startScavengerHunt() {
        let challenges = buildSessionChallenges()
        guard !challenges.isEmpty else {
            print("[Hunt] No eligible challenges — staying in explore mode")
            mode = .exploring
            return
        }
        quizSession = QuizSession(challenges: challenges)
        mode = .quizPrompting
        print("[Hunt] Starting session with \(challenges.count) challenges: \(challenges.map(\.displayText))")
        speakCurrentQuizWord()
    }

    /// Build a mixed session of word and category challenges.
    private func buildSessionChallenges() -> [Challenge] {
        let total = QuizSession.challengesPerSession
        let cats = selectedCategories.isEmpty ? nil : selectedCategories
        let words = wordProgressStore.selectQuizWords(count: total, location: selectedLocation, categories: cats)
        guard !words.isEmpty else { return [] }

        // Pick 1-2 category challenges from eligible categories
        let eligibleCategories = availableCategoriesForSession()
        let categoryCount = eligibleCategories.isEmpty ? 0 : Int.random(in: 1...min(2, eligibleCategories.count))
        let pickedCategories = Array(eligibleCategories.shuffled().prefix(categoryCount))

        var challenges: [Challenge] = []

        for category in pickedCategories {
            challenges.append(Challenge(target: .category(category)))
        }

        // Fill remaining with word challenges (skip words that belong to picked categories
        // to avoid "Find an animal" followed by "Find the dog")
        let categoryWords = Set(pickedCategories.flatMap { WordProgressStore.quizCategories[$0] ?? [] })
        let filteredWords = words.filter { !categoryWords.contains($0) }

        let wordCount = total - challenges.count
        for word in filteredWords.prefix(wordCount) {
            challenges.append(Challenge(target: .word(word)))
        }

        if challenges.count < total {
            for word in words where challenges.count < total {
                if !challenges.contains(where: { $0.displayText == word }) {
                    challenges.append(Challenge(target: .word(word)))
                }
            }
        }

        return challenges.shuffled()
    }

    /// Categories eligible for this session, filtered by location and parent selection.
    private func availableCategoriesForSession() -> [String] {
        let quizzable = Set(wordProgressStore.quizEligibleWords())
        let parentCats = selectedCategories

        return WordProgressStore.quizCategories.compactMap { category, words in
            // If parent selected categories, only include those
            if !parentCats.isEmpty && !parentCats.contains(category) { return nil }

            // Need 3+ quizzable words in this category at the selected location
            let matching = words.filter { word in
                guard quizzable.contains(word) else { return false }
                guard let loc = selectedLocation else { return true }
                let wordLocs = WordProgressStore.wordLocations[word] ?? []
                return wordLocs.contains(loc)
            }
            return matching.count >= 3 ? category : nil
        }
    }

    /// Switch from scavenger hunt to free-roam explore mode.
    func switchToExplore() {
        quizSession = nil
        wordSpeaker.stop()
        mode = .exploring
    }

    private func speakCurrentQuizWord() {
        guard let challenge = quizSession?.currentChallenge else { return }
        // Small delay so the UI settles before audio plays
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            switch challenge.target {
            case .word(let word):
                self?.wordSpeaker.speakHuntPrompt(word: word, onComplete: {})
            case .category(let category):
                self?.wordSpeaker.speakCategoryPrompt(category: category, onComplete: {})
            }
        }
    }

    /// Computes a normalized crop rect centered on the tap point.
    /// Uses cropFraction of the smaller frame dimension as the crop size.
    private func tapCenteredCropRect(tapPoint: CGPoint, cropFraction: CGFloat, bufferWidth: Int, bufferHeight: Int) -> CGRect {
        let w = CGFloat(bufferWidth)
        let h = CGFloat(bufferHeight)
        let minDim = min(w, h)
        let cropPixels = minDim * cropFraction

        // Normalized crop size
        let cropW = cropPixels / w
        let cropH = cropPixels / h

        var x = tapPoint.x - cropW / 2
        var y = tapPoint.y - cropH / 2

        // Clamp to [0, 1] bounds
        x = max(0, min(x, 1.0 - cropW))
        y = max(0, min(y, 1.0 - cropH))

        return CGRect(x: x, y: y, width: cropW, height: cropH)
    }

    private func cropPixelBuffer(_ buffer: CVPixelBuffer, to normalizedRect: CGRect) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)

        // CIImage uses bottom-left origin. normalizedRect uses top-left origin.
        // Flip Y: ciY = bufferHeight - topLeftY - cropHeight
        let pxX = normalizedRect.origin.x * CGFloat(width)
        let pxW = normalizedRect.width * CGFloat(width)
        let pxH = normalizedRect.height * CGFloat(height)
        let pxY = CGFloat(height) - normalizedRect.origin.y * CGFloat(height) - pxH

        let cropRect = CGRect(x: pxX, y: pxY, width: pxW, height: pxH).integral

        guard cropRect.width > 0, cropRect.height > 0 else { return nil }

        // Crop, then translate extent to (0,0) so CIContext.render intersects the output buffer
        let cropped = CIImage(cvPixelBuffer: buffer)
            .cropped(to: cropRect)
            .transformed(by: CGAffineTransform(translationX: -cropRect.origin.x, y: -cropRect.origin.y))

        var croppedBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(cropRect.width), Int(cropRect.height),
                           kCVPixelFormatType_32BGRA, nil, &croppedBuffer)
        guard let output = croppedBuffer else { return nil }
        sharedCIContext.render(cropped, to: output)
        return output
    }

}

extension AppState: CameraManagerDelegate {
    nonisolated func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        let bufW = CVPixelBufferGetWidth(pixelBuffer)
        let bufH = CVPixelBufferGetHeight(pixelBuffer)
        let timeSeconds = CMTimeGetSeconds(timestamp)

        let isLandscape = bufW > bufH
        Task { @MainActor [isLandscape] in
            if !self.hasLoggedBufferDims {
                self.hasLoggedBufferDims = true
                print("[Camera] Buffer dimensions: \(bufW)×\(bufH) (\(isLandscape ? "LANDSCAPE" : "PORTRAIT"))")
            }
            self.bufferIsLandscape = isLandscape
        }

        // Skip segmentation when idle to save battery
        let lastTap = _lastTapTime.withLock { $0 }
        if Date().timeIntervalSince(lastTap) > idleTimeout { return }

        // Exploring mode: run segmentation (~3fps)
        let result = segmentationEngine.segment(pixelBuffer: pixelBuffer)

        if !result.instances.isEmpty {
            print("[Seg] Found \(result.instances.count) instances at t=\(String(format: "%.1f", timeSeconds))")
        }

        Task { @MainActor in
            if mode == .exploring || mode == .quizPrompting {
                instanceTracker.update(instances: result.instances, timestamp: timeSeconds)
                liveSegmentation = result
            }
        }
    }
}
