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
///
/// Skip = remove + replace: skipping removes the current challenge and draws a fresh
/// replacement from the pool so the child always gets the same number of real attempts.
/// Back = undo skip: restores the skipped challenge and removes the replacement.
@MainActor
final class QuizSession: ObservableObject {
    var challenges: [Challenge]
    @Published var currentIndex: Int = 0
    @Published var attemptsOnCurrent: Int = 0
    @Published var lastResult: QuizAnswerResult?
    @Published var results: [(challenge: Challenge, correct: Bool, foundWord: String?)] = []

    /// The word found on correct category challenges (shown briefly after match).
    @Published var lastFoundWord: String?

    /// Undo stack for skips. Each entry records the original challenge and the index it was at.
    var skipStack: [(original: Challenge, index: Int)] = []

    /// All words skipped this session (prevents recycling into replacements).
    var skippedWords: Set<String> = []

    static let maxAttempts = 3
    static let challengesPerSession = 5

    /// Convenience: all display words for dot tracking.
    var words: [String] { challenges.map(\.displayText) }

    var currentChallenge: Challenge? {
        guard currentIndex < challenges.count else { return nil }
        return challenges[currentIndex]
    }

    /// The text to display for the current challenge.
    var currentWord: String? { currentChallenge?.displayText }

    var isComplete: Bool { currentIndex >= challenges.count }
    var correctCount: Int { results.filter(\.correct).count }

    /// End the session immediately (e.g., pool exhausted on last challenge).
    func forceComplete() {
        currentIndex = challenges.count
    }

    /// Can go back if there's a skip to undo, or a previous non-correct challenge to revisit.
    var canGoBack: Bool {
        if !skipStack.isEmpty { return true }
        // Check if any previous challenge was NOT correctly answered
        for i in (0..<currentIndex).reversed() {
            if let r = result(at: i), r.correct { continue }
            return true
        }
        return false
    }

    /// Look up the result for a specific challenge index (nil if not yet attempted).
    func result(at index: Int) -> (correct: Bool, foundWord: String?)? {
        guard index < challenges.count else { return nil }
        let challenge = challenges[index]
        return results.first(where: { $0.challenge == challenge }).map { ($0.correct, $0.foundWord) }
    }

    init(challenges: [Challenge]) {
        self.challenges = challenges
    }

    /// Convenience init for word-only sessions (backwards compat).
    convenience init(words: [String]) {
        self.init(challenges: words.map { Challenge(target: .word($0)) })
    }

    func recordResult(correct: Bool, foundWord: String? = nil) {
        guard let challenge = currentChallenge else { return }
        // Replace existing result for this challenge (prevents duplicates on re-attempt)
        results.removeAll { $0.challenge == challenge }
        results.append((challenge: challenge, correct: correct, foundWord: foundWord))
        lastFoundWord = correct ? foundWord : nil
    }

    func advance() {
        attemptsOnCurrent = 0
        lastResult = nil
        lastFoundWord = nil
        currentIndex += 1
    }

    /// Skip the current challenge: remove it, append a replacement (if provided).
    /// Returns the removed challenge for undo tracking.
    @discardableResult
    func skip(replacement: Challenge?) -> Challenge {
        let removed = challenges.remove(at: currentIndex)
        skippedWords.insert(removed.displayText)
        skipStack.append((original: removed, index: currentIndex))

        if let replacement {
            challenges.append(replacement)
        }
        // currentIndex now points to the next challenge (since we removed one)
        // If we're past the end, don't advance further
        if currentIndex > challenges.count {
            currentIndex = challenges.count
        }

        attemptsOnCurrent = 0
        lastResult = nil
        lastFoundWord = nil
        return removed
    }

    /// Undo the last skip: restore the original challenge, remove the replacement.
    func undoSkip() {
        guard let entry = skipStack.popLast() else { return }
        skippedWords.remove(entry.original.displayText)

        // Remove the replacement (last element, added during skip)
        if challenges.count > entry.index {
            challenges.removeLast()
        }

        // Re-insert the original at its old position
        challenges.insert(entry.original, at: entry.index)
        currentIndex = entry.index

        attemptsOnCurrent = 0
        lastResult = nil
        lastFoundWord = nil
    }

    func goBack() {
        guard canGoBack else { return }

        // If last skip was at or after current position, undo it
        if let lastSkip = skipStack.last, lastSkip.index <= currentIndex {
            undoSkip()
            return
        }

        // Go back to nearest previous non-correct challenge
        var targetIndex = currentIndex - 1
        while targetIndex >= 0 {
            if let r = result(at: targetIndex), r.correct {
                targetIndex -= 1
            } else {
                break
            }
        }
        guard targetIndex >= 0 else { return }

        // Remove the wrong result for the target challenge so it can be re-attempted
        let challenge = challenges[targetIndex]
        results.removeAll { $0.challenge == challenge }

        currentIndex = targetIndex
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
    @Published var showTapRipple: Bool = false
    private var rippleDismissTask: Task<Void, Never>?

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

    /// Triggers a brief ripple animation at tapScreenPoint, auto-dismissed after 0.5s.
    private func triggerTapRipple() {
        rippleDismissTask?.cancel()
        showTapRipple = true
        rippleDismissTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled else { return }
            showTapRipple = false
        }
    }

    func handleTap(imagePoint: CGPoint, screenPoint: CGPoint) {
        guard mode == .exploring else { return }

        _lastTapTime.withLock { $0 = Date() }

        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()

        tapScreenPoint = screenPoint
        triggerTapRipple()
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

        let cropRect = instanceAwareCropRect(tapPoint: bufferPoint, bufferWidth: bufW, bufferHeight: bufH)
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
                // Unrecognized — haptic + gentle audio feedback, then return to exploring.
                print("[Tap] Unrecognized object — returning to exploring")
                let errorGenerator = UINotificationFeedbackGenerator()
                errorGenerator.notificationOccurred(.warning)
                wordSpeaker.speakUnrecognized {
                    // Audio done, no state change needed
                }
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
        triggerTapRipple()
        mode = .quizClassifying

        guard let segResult = liveSegmentation else {
            print("[Quiz] No segmentation data")
            mode = .quizPrompting
            return
        }

        let bufW = CVPixelBufferGetWidth(segResult.pixelBuffer)
        let bufH = CVPixelBufferGetHeight(segResult.pixelBuffer)
        let bufferPoint = CGPoint(x: 1.0 - imagePoint.y, y: imagePoint.x)
        let cropRect = instanceAwareCropRect(tapPoint: bufferPoint, bufferWidth: bufW, bufferHeight: bufH)

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

                let advanceAfterCelebration: () -> Void = { [weak self] in
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                        guard let self, self.quizGeneration == gen else { return }
                        self.advanceQuiz()
                    }
                }

                // Category challenges: name the found word (new info for the child).
                // Word challenges: celebration only (child already knows the word from the prompt).
                if case .category = challenge.target {
                    wordSpeaker.speakCelebration(word: foundWord, onComplete: advanceAfterCelebration)
                } else {
                    wordSpeaker.speakCelebrationOnly(onComplete: advanceAfterCelebration)
                }
            } else {
                // Wrong — shake + haptic + encouragement audio, then retry or advance.
                session.attemptsOnCurrent += 1
                session.lastResult = .wrong(actual: result.word)
                let errorGenerator = UINotificationFeedbackGenerator()
                errorGenerator.notificationOccurred(.error)
                print("[Quiz] Wrong — expected '\(challenge.displayText)', got '\(result.word ?? "nil")'")

                let hasRetries = session.attemptsOnCurrent < QuizSession.maxAttempts

                // Play encouragement audio, then retry or advance.
                wordSpeaker.speakEncouragement { [weak self] in
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
            print("[Quiz] Session complete: \(session.correctCount)/\(session.results.count)")
        } else {
            mode = .quizPrompting
            speakCurrentQuizWord()
        }
    }

    /// Skip the current quiz word (target not in room).
    /// Removes the challenge and draws a fresh replacement so the child still gets the same
    /// number of real attempts. Child can press back to undo the skip.
    func skipQuizWord() {
        guard let session = quizSession else { return }
        wordSpeaker.stop()
        quizGeneration += 1

        guard session.currentIndex < session.challenges.count else { return }

        // Draw a replacement challenge (excluding current session words + all skipped words)
        let replacement = drawReplacementChallenge(session: session)

        // Last challenge with no replacement: end the session rather than trapping the user.
        if session.challenges.count <= 1 && replacement == nil {
            print("[Quiz] Skipping last challenge — pool exhausted, ending session")
            if case .word(let w) = session.challenges[session.currentIndex].target {
                wordProgressStore.recordQuizSkip(word: w)
            }
            session.forceComplete()
            wordSpeaker.stop()
            mode = .quizResult
            return
        }

        let removed = session.skip(replacement: replacement)
        let word = removed.displayText
        print("[Quiz] Skipped '\(word)' → replaced with '\(replacement?.displayText ?? "none")' (\(session.challenges.count) challenges)")

        // Update cooldown so this word doesn't dominate future sessions
        if case .word(let w) = removed.target {
            wordProgressStore.recordQuizSkip(word: w)
        }

        if session.isComplete {
            wordSpeaker.stop()
            print("[Quiz] Session complete: \(session.correctCount)/\(session.results.count)")
        } else {
            mode = .quizPrompting
            speakCurrentQuizWord()
        }
    }

    /// Draw a replacement challenge for a skipped word.
    /// Excludes all current session words and previously skipped words.
    private func drawReplacementChallenge(session: QuizSession) -> Challenge? {
        let currentWords = Set(session.challenges.map(\.displayText))
        let excluded = currentWords.union(session.skippedWords)

        let cats = selectedCategories.isEmpty ? nil : selectedCategories
        let candidates = wordProgressStore.selectQuizWords(
            count: 20, location: selectedLocation, categories: cats
        ).filter { !excluded.contains($0) }

        guard let word = candidates.first else {
            print("[Quiz] No replacement available — pool exhausted")
            return nil
        }
        return Challenge(target: .word(word))
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

    /// Go back to the previous quiz word, or undo the last skip.
    func goBackQuizWord() {
        guard let session = quizSession, session.canGoBack else { return }
        wordSpeaker.stop()
        let wasSameIndex = session.currentIndex
        session.goBack()
        let action = session.currentIndex == wasSameIndex ? "undo skip" : "go back"
        print("[Quiz] \(action) → now at '\(session.currentChallenge?.displayText ?? "?")'")
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

    /// Computes a tap-centered square crop for classification.
    /// Uses instance bounding box to inform crop size (so large objects get a bigger crop),
    /// but always centers on the tap point and caps the size so the model sees the object,
    /// not the entire room.
    ///
    /// Crop size logic:
    /// - No instance: fixed 270px crop (25% of 1080)
    /// - Instance bbox small: use bbox's larger dimension + 20% padding
    /// - Instance bbox large: cap at 500px (46% of 1080) so the object stays dominant
    private func instanceAwareCropRect(tapPoint: CGPoint, bufferWidth: Int, bufferHeight: Int) -> CGRect {
        let w = CGFloat(bufferWidth)
        let h = CGFloat(bufferHeight)
        let minDim = min(w, h)

        let minCropFraction: CGFloat = 0.25   // ~270px on 1080 buffer
        let maxCropFraction: CGFloat = 0.46    // ~500px on 1080 buffer

        var cropFraction = minCropFraction

        if let instance = instanceTracker.instance(at: tapPoint) {
            let bbox = instance.boundingBox
            // Use the larger bbox dimension (in pixels) to set crop size
            let bboxPxW = bbox.width * w
            let bboxPxH = bbox.height * h
            let bboxMaxPx = max(bboxPxW, bboxPxH)
            // Add 20% padding around the bbox dimension
            let desiredPx = bboxMaxPx * 1.2
            let desiredFraction = desiredPx / minDim
            // Clamp between min and max
            cropFraction = min(max(desiredFraction, minCropFraction), maxCropFraction)
            let cropPx = Int(cropFraction * minDim)
            print("[Crop] Instance bbox=\(bbox) (px: \(Int(bboxPxW))x\(Int(bboxPxH))) → tap-centered \(cropPx)x\(cropPx)px crop")
        } else {
            print("[Crop] No instance at tap point — using fixed \(Int(minCropFraction * minDim))px crop")
        }

        return tapCenteredCropRect(tapPoint: tapPoint, cropFraction: cropFraction,
                                   bufferWidth: bufferWidth, bufferHeight: bufferHeight)
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
