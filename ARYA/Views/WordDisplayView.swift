import SwiftUI

struct WordDisplayView: View {
    let word: String
    let letterStates: [LetterState]
    var fontSize: CGFloat = 80
    /// Color for letters in .upcoming state (default: dim white for learning overlay).
    var upcomingColor: Color = .white.opacity(0.3)

    /// Split multi-word strings into lines, keep single words whole.
    private var lines: [String] {
        let parts = word.split(separator: " ").map(String.init)
        return parts.count > 1 ? parts : [word]
    }

    /// Font size per line. Full size for most words, scaled for 9+ char single-word lines.
    private func lineFontSize(_ line: String) -> CGFloat {
        let count = line.count
        if count <= 8 { return fontSize }
        if count <= 10 { return fontSize * 0.85 }
        return fontSize * 0.72
    }

    /// Running character offset for a given line index (accounts for spaces between words).
    private func charOffset(forLine lineIndex: Int) -> Int {
        let parts = word.split(separator: " ").map(String.init)
        var offset = 0
        for i in 0..<lineIndex {
            offset += parts[i].count + 1 // +1 for the space
        }
        return offset
    }

    var body: some View {
        VStack(spacing: lines.count > 1 ? 4 : 0) {
            ForEach(Array(lines.enumerated()), id: \.offset) { lineIndex, line in
                let size = lineFontSize(line)
                let baseOffset = charOffset(forLine: lineIndex)
                HStack(spacing: size * 0.18) {
                    ForEach(Array(line.enumerated()), id: \.offset) { charIndex, char in
                        let globalIndex = baseOffset + charIndex
                        Text(String(char))
                            .font(.system(size: size, weight: .bold, design: .rounded))
                            .foregroundColor(color(for: letterState(at: globalIndex)))
                            .shadow(color: glowColor(for: letterState(at: globalIndex)), radius: size * 0.25)
                            .animation(nil, value: letterState(at: globalIndex))
                    }
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
        case .upcoming: return upcomingColor
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
