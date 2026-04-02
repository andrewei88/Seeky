import SwiftUI

/// Scavenger hunt overlay on top of the camera feed.
/// No separate result screen — feedback is shown inline via dot animations,
/// word bounce/shake, and audio. This keeps the child focused on the camera view.
struct QuizOverlayView: View {
    @ObservedObject var session: QuizSession
    @ObservedObject var wordSpeaker: WordSpeaker
    let mode: AppMode
    let onSkip: () -> Void
    let onGoBack: () -> Void
    let onRetry: () -> Void
    let onAdvance: () -> Void
    let onOverride: () -> Void
    let onNewHunt: () -> Void
    let onExplore: () -> Void
    let onReplay: () -> Void

    // Animation state
    @State private var wordBounceY: CGFloat = 0
    @State private var wordShakeOffset: CGFloat = 0
    @State private var celebratingIndex: Int? = nil

    // Letter highlight state
    @State private var isCelebrating: Bool = false
    /// Tracks which word's audio has finished playing (nil = no word spoken yet this challenge).
    /// Using the word string instead of a boolean so that skips (which replace the challenge
    /// at the same index) automatically invalidate the state.
    @State private var spokenWord: String? = nil
    /// Cached timing data to avoid disk I/O on every frame render.
    @State private var cachedTiming: TimingData? = nil
    @State private var cachedTimingWord: String? = nil

    var body: some View {
        if session.isComplete {
            completionView
        } else {
            promptView
        }
    }

    // MARK: - Prompt (unified view for prompting + result feedback)

    private var promptView: some View {
        VStack(spacing: 0) {
            // Progress dots
            HStack(spacing: 8) {
                ForEach(0..<session.words.count, id: \.self) { i in
                    Circle()
                        .fill(dotColor(for: i))
                        .frame(width: dotSize(for: i), height: dotSize(for: i))
                        .shadow(color: dotGlow(for: i), radius: 6)
                        .animation(.spring(response: 0.4, dampingFraction: 0.6), value: celebratingIndex)
                }
            }
            .padding(.top, 70)

            // Target prompt
            if let challenge = session.currentChallenge {
                VStack(spacing: 4) {
                    // Unified display: both word and category challenges show just the name
                    WordDisplayView(
                        word: challenge.displayText,
                        letterStates: currentLetterStates(for: challenge.displayText),
                        fontSize: 56,
                        upcomingColor: .white
                    )
                    .shadow(color: .black.opacity(0.5), radius: 4, y: 2)

                    // Category: show the found word after correct match
                    if case .category = challenge.target,
                       let found = session.lastFoundWord,
                       session.lastResult == .correct {
                        Text(found)
                            .font(.system(size: 28, weight: .semibold, design: .rounded))
                            .foregroundColor(.green)
                            .transition(.scale.combined(with: .opacity))
                    }

                    // Hint after 2+ failed attempts on word challenges
                    if case .word(let targetWord) = challenge.target,
                       session.attemptsOnCurrent >= 2,
                       mode == .quizPrompting {
                        let hint = String(targetWord.prefix(1)).uppercased()
                        Text("Hint: starts with \(hint)")
                            .font(.system(size: 18, weight: .medium, design: .rounded))
                            .foregroundColor(.yellow.opacity(0.8))
                            .transition(.opacity)
                    }
                }
                .padding(.top, 12)
                .offset(x: wordShakeOffset, y: wordBounceY)

                // Replay speaker icon (hidden during celebration/result)
                if mode == .quizPrompting {
                    Button { onReplay() } label: {
                        Image(systemName: "speaker.wave.2.fill")
                            .font(.system(size: 18))
                            .foregroundColor(.white.opacity(0.4))
                            .frame(width: 44, height: 36)
                    }
                    .padding(.top, 4)
                }
            }

            Spacer()

            // Bottom bar: pencil (left) + nav arrows (right)
            HStack {
                // Parent override pencil (always available during result)
                if mode == .quizResult {
                    Button { onOverride() } label: {
                        Image(systemName: "pencil.circle.fill")
                            .font(.system(size: 32))
                            .foregroundColor(.white.opacity(0.25))
                    }
                } else {
                    // Back arrow (hidden on first word)
                    Button { onGoBack() } label: {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 22, weight: .medium))
                            .foregroundColor(.white.opacity(0.4))
                            .frame(width: 56, height: 56)
                    }
                    .opacity(session.canGoBack ? 1 : 0)
                    .disabled(!session.canGoBack)
                }

                Spacer()

                if mode == .quizResult {
                    // Continue arrow (fallback when auto-advance callback is dropped)
                    Button { onAdvance() } label: {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 22, weight: .medium))
                            .foregroundColor(.white.opacity(0.4))
                            .frame(width: 56, height: 56)
                    }
                } else {
                    // Skip arrow (visible in prompting AND classifying so user is never stuck)
                    Button { onSkip() } label: {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 22, weight: .medium))
                            .foregroundColor(.white.opacity(0.4))
                            .frame(width: 56, height: 56)
                    }
                }
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 44)
        }
        .onChange(of: session.lastResult) { _, newResult in
            handleResultChange(newResult)
        }
        .onChange(of: wordSpeaker.isPlayingWord) { oldValue, newValue in
            // Word audio finished → record which word was spoken.
            // Uses lastPlayedLabel (stable) instead of session.currentWord
            // (which may have changed by the time this deferred callback fires).
            if oldValue && !newValue {
                spokenWord = wordSpeaker.lastPlayedLabel
            }
        }
        .onChange(of: session.currentWord) { _, _ in
            // New challenge (advance, skip, or goBack) — reset highlight state
            isCelebrating = false
            spokenWord = nil
        }
    }

    // MARK: - Completion

    private var completionView: some View {
        VStack(spacing: 24) {
            Spacer()

            Text(completionTitle)
                .font(.system(size: 40, weight: .bold, design: .rounded))
                .foregroundColor(.white)

            Text("\(session.correctCount) out of \(session.challenges.count)")
                .font(.system(size: 24, design: .rounded))
                .foregroundColor(.white.opacity(0.7))

            // Word-by-word results (in challenge order)
            VStack(spacing: 8) {
                ForEach(0..<session.challenges.count, id: \.self) { i in
                    let challenge = session.challenges[i]
                    let isCorrect = session.result(at: i)?.correct ?? false
                    HStack(spacing: 10) {
                        if isCorrect {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                                .font(.system(size: 18))
                        } else {
                            Circle()
                                .fill(Color.white.opacity(0.15))
                                .frame(width: 18, height: 18)
                        }
                        Text(challenge.displayText)
                            .font(.system(size: 20, weight: .medium, design: .rounded))
                            .foregroundColor(.white)
                        Spacer()
                    }
                }
            }
            .padding(.horizontal, 60)

            Spacer()

            VStack(spacing: 12) {
                Button { onNewHunt() } label: {
                    Text("New Hunt")
                        .font(.system(size: 20, weight: .semibold, design: .rounded))
                        .foregroundColor(.white)
                        .padding(.horizontal, 48)
                        .padding(.vertical, 14)
                        .background(Color.blue.opacity(0.6))
                        .cornerRadius(28)
                }

                Button { onExplore() } label: {
                    Text("Free Explore")
                        .font(.system(size: 16, weight: .medium, design: .rounded))
                        .foregroundColor(.white.opacity(0.4))
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                }
            }
            .padding(.bottom, 80)
        }
    }

    // MARK: - Animations

    private func handleResultChange(_ result: QuizAnswerResult?) {
        guard let result else { return }

        if case .correct = result {
            print("[Animation] Correct! Triggering bounce + golden flash at index \(session.currentIndex)")
            celebratingIndex = session.currentIndex
            isCelebrating = true

            // Word jumps up — pure vertical movement, no scaling.
            // Spring with low damping (0.3) overshoots, creating a physical bounce.
            withAnimation(.spring(response: 0.3, dampingFraction: 0.3)) {
                wordBounceY = -50
            }
            // Spring back down after holding at peak
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.35)) {
                    wordBounceY = 0
                }
            }
            // End golden flash after 1s, settle to dim yellow
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                withAnimation(.easeOut(duration: 0.3)) {
                    isCelebrating = false
                }
            }
        } else {
            // Word shake: quick horizontal wiggle
            let shakeSequence: [(CGFloat, Double)] = [
                (10, 0.0), (-8, 0.06), (6, 0.12), (-3, 0.18), (0, 0.24)
            ]
            for (offset, delay) in shakeSequence {
                withAnimation(.easeInOut(duration: 0.06).delay(delay)) {
                    wordShakeOffset = offset
                }
            }
        }
    }

    // MARK: - Letter states

    private func currentLetterStates(for word: String) -> [LetterState] {
        // Celebration flash: all letters bright gold with glow
        if isCelebrating {
            return word.map { $0 == " " ? .space : .active }
        }

        // Live highlighting during word audio playback
        if wordSpeaker.isPlayingWord,
           let timing = loadCachedTiming(for: word) {
            let highlighter = LetterHighlighter(timing: timing)
            return highlighter.letterStates(at: wordSpeaker.currentTime)
        }

        // After pronunciation: letters stay dim yellow (only for the current word)
        if spokenWord == word {
            return word.map { $0 == " " ? .space : .spoken }
        }

        // Default: white (word hasn't been spoken yet)
        return word.map { $0 == " " ? .space : .upcoming }
    }

    /// Load timing data from cache, only hitting disk when the word changes.
    private func loadCachedTiming(for word: String) -> TimingData? {
        if cachedTimingWord == word { return cachedTiming }
        // Word changed — load from disk once and cache
        let timing = TimingData.load(word: word)
        DispatchQueue.main.async {
            cachedTimingWord = word
            cachedTiming = timing
        }
        return timing
    }

    // MARK: - Dot styling

    private func dotColor(for index: Int) -> Color {
        if let result = session.result(at: index) {
            return result.correct ? .green : .white.opacity(0.15)
        }
        if index == session.currentIndex {
            if celebratingIndex == index { return .green }
            return .white
        }
        return .white.opacity(0.3)
    }

    private func dotSize(for index: Int) -> CGFloat {
        if index == celebratingIndex && index == session.currentIndex {
            return 14
        }
        if let result = session.result(at: index), result.correct {
            return 10
        }
        return 8
    }

    private func dotGlow(for index: Int) -> Color {
        if index == celebratingIndex && index == session.currentIndex {
            return .green.opacity(0.6)
        }
        return .clear
    }

    private var completionTitle: String {
        let total = session.challenges.count
        let ratio = total == 0 ? 0 : Double(session.correctCount) / Double(total)
        if ratio >= 1.0 { return "Perfect!" }
        if ratio >= 0.6 { return "Great job!" }
        if ratio > 0 { return "Good try!" }
        return "Keep going!"
    }

}
