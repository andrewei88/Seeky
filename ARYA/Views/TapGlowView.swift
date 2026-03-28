import SwiftUI

/// Brief concentric ripple at the tap point.
/// Expands outward and fades in ~0.5s. Confirms "I heard your tap, right here"
/// without obscuring the object or competing with the word label.
struct TapRippleView: View {
    let screenPoint: CGPoint

    @State private var scale: CGFloat = 0.3
    @State private var opacity: Double = 0.6

    var body: some View {
        Circle()
            .strokeBorder(Color.white.opacity(0.8), lineWidth: 2.5)
            .frame(width: 80, height: 80)
            .scaleEffect(scale)
            .opacity(opacity)
            .position(screenPoint)
            .onAppear {
                withAnimation(.easeOut(duration: 0.45)) {
                    scale = 1.2
                    opacity = 0.0
                }
            }
    }
}
