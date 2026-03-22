import SwiftUI

/// A soft pulsing glow circle at the tap point.
/// Simple visual indicator for children — no bounding box, no frozen frame.
struct TapGlowView: View {
    let screenPoint: CGPoint // Normalized 0-1

    @State private var pulse: CGFloat = 0.8
    @State private var opacity: Double = 0.7

    var body: some View {
        GeometryReader { geo in
            let center = CGPoint(
                x: screenPoint.x * geo.size.width,
                y: screenPoint.y * geo.size.height
            )

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
                .position(center)
                .animation(
                    .easeInOut(duration: 1.0).repeatForever(autoreverses: true),
                    value: pulse
                )
                .onAppear {
                    pulse = 1.1
                    opacity = 0.9
                }
        }
        .ignoresSafeArea()
    }
}
