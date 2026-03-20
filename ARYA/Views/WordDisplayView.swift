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
