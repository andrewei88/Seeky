import SwiftUI

/// A soft pulsing glow circle at the tap point.
/// Simple visual indicator for children — no bounding box, no frozen frame.
struct TapGlowView: View {
    let screenPoint: CGPoint // Raw screen coordinates (points)

    @State private var pulse: CGFloat = 0.8
    @State private var opacity: Double = 0.7

    var body: some View {
        Circle()
            .fill(
                RadialGradient(
                    gradient: Gradient(colors: [
                        Color.white.opacity(0.5),
                        Color.yellow.opacity(0.3),
                        Color.yellow.opacity(0.0)
                    ]),
                    center: .center,
                    startRadius: 10,
                    endRadius: 80
                )
            )
            .frame(width: 160, height: 160)
            .scaleEffect(pulse)
            .opacity(opacity)
            .position(screenPoint)
            .animation(
                .easeInOut(duration: 1.0).repeatForever(autoreverses: true),
                value: pulse
            )
            .onAppear {
                pulse = 1.1
                opacity = 0.9
            }
    }
}
