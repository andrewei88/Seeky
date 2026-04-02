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
    private let allReferencedIndices: Set<Int>

    init(timing: TimingData) {
        self.timing = timing
        self.letterCount = timing.word.count
        self.allReferencedIndices = Set(timing.phonemes.flatMap(\.letters))
    }

    func letterStates(at time: Double) -> [LetterState] {
        let activeIndices = Set(timing.activeLetterIndices(at: time))
        let spokenIndices = Set(timing.spokenLetterIndices(at: time))

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
