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
    @State private var wordBounce: CGFloat = 1.0
    @State private var wordShakeOffset: CGFloat = 0
    @State private var celebratingIndex: Int? = nil

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
                    if case .category(let category) = challenge.target {
                        // Category challenge: show "Find a..." label + category name
                        Text(categoryArticle(category))
                            .font(.system(size: 20, weight: .medium, design: .rounded))
                            .foregroundColor(.white.opacity(0.5))
                            .padding(.top, 8)

                        Text(category)
                            .font(.system(size: 56, weight: .bold, design: .rounded))
                            .foregroundColor(.white)
                            .shadow(color: .black.opacity(0.5), radius: 4, y: 2)

                        // Show the found word after correct match
                        if let found = session.lastFoundWord, session.lastResult == .correct {
                            Text(found)
                                .font(.system(size: 28, weight: .semibold, design: .rounded))
                                .foregroundColor(.green)
                                .transition(.scale.combined(with: .opacity))
                        }
                    } else {
                        // Word challenge: letter-by-letter highlighting
                        WordDisplayView(
                            word: challenge.displayText,
                            letterStates: currentLetterStates(for: challenge.displayText),
                            fontSize: 56,
                            upcomingColor: .white
                        )
                        .shadow(color: .black.opacity(0.5), radius: 4, y: 2)
                    }
                }
                .padding(.top, 12)
                .scaleEffect(wordBounce)
                .offset(x: wordShakeOffset)

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

                if mode == .quizPrompting {
                    // Skip arrow
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
    }

    // MARK: - Completion

    private var completionView: some View {
        VStack(spacing: 24) {
            Spacer()

            Text("Great job!")
                .font(.system(size: 40, weight: .bold, design: .rounded))
                .foregroundColor(.white)

            Text("\(session.correctCount) out of \(session.results.count)")
                .font(.system(size: 24, design: .rounded))
                .foregroundColor(.white.opacity(0.7))

            HStack(spacing: 8) {
                ForEach(0..<session.results.count, id: \.self) { i in
                    Circle()
                        .fill(session.results[i].correct ? Color.green : Color.white.opacity(0.2))
                        .frame(width: session.results[i].correct ? 12 : 10,
                               height: session.results[i].correct ? 12 : 10)
                }
            }

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
            // Word bounce: 1.0 → 1.15 → 1.0
            withAnimation(.spring(response: 0.35, dampingFraction: 0.5)) {
                wordBounce = 1.15
                celebratingIndex = session.currentIndex
            }
            withAnimation(.spring(response: 0.3, dampingFraction: 0.7).delay(0.35)) {
                wordBounce = 1.0
            }
        } else {
            // Word shake: quick horizontal wiggle
            let shakeSequence: [(CGFloat, Double)] = [
                (8, 0.0), (-6, 0.06), (4, 0.12), (-2, 0.18), (0, 0.24)
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
        guard wordSpeaker.isPlayingWord,
              let timing = TimingData.load(word: word) else {
            // Not playing word audio — all letters white (upcoming with full-white color)
            return word.map { $0 == " " ? .space : .upcoming }
        }
        let highlighter = LetterHighlighter(timing: timing)
        return highlighter.letterStates(at: wordSpeaker.currentTime)
    }

    // MARK: - Dot styling

    private func dotColor(for index: Int) -> Color {
        if index < session.results.count {
            // Completed word: green if correct, empty/dim if wrong
            return session.results[index].correct ? .green : .white.opacity(0.15)
        } else if index == session.currentIndex {
            // Currently celebrating correct answer
            if celebratingIndex == index { return .green }
            return .white
        }
        return .white.opacity(0.3)
    }

    private func dotSize(for index: Int) -> CGFloat {
        // Celebrating dot grows briefly
        if index == celebratingIndex && index == session.currentIndex {
            return 14
        }
        if index < session.results.count && session.results[index].correct {
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

    /// Returns "Find a" or "Find an" depending on the category name.
    private func categoryArticle(_ category: String) -> String {
        let vowels: Set<Character> = ["a", "e", "i", "o", "u"]
        if let first = category.lowercased().first, vowels.contains(first) {
            return "Find an"
        }
        return "Find a"
    }
}
