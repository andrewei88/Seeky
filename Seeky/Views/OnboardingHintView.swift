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
