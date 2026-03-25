import SwiftUI

struct LearningOverlayView: View {
    let word: String
    @ObservedObject var wordSpeaker: WordSpeaker

    @State private var letterHighlighter: LetterHighlighter?
    @State private var hasStartedSpeaking = false

    var body: some View {
        GeometryReader { geo in
            VStack {
                Spacer()

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
            .onAppear {
                if let timing = TimingData.load(word: word) {
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
}
