import SwiftUI

/// Dims the screen except for a rounded rectangle around the detected object's bounding box.
/// Uses the object-tracking bounding box so it follows the object in real time.
struct SpotlightOverlayView: View {
    let boundingBox: CGRect  // Normalized image coordinates (0-1)
    let bufferIsLandscape: Bool
    @State private var pulseAmount: CGFloat = 0

    private let padding: CGFloat = 16
    private let cornerRadius: CGFloat = 16

    var body: some View {
        GeometryReader { geo in
            let screenRect = convertToScreen(bbox: boundingBox, in: geo.size)

            Canvas { context, size in
                let fullRect = CGRect(origin: .zero, size: size)
                var dimPath = Path(fullRect)

                // Cut out the object area with padding and pulse
                let highlightRect = screenRect.insetBy(
                    dx: -(padding + pulseAmount * 4),
                    dy: -(padding + pulseAmount * 4)
                )
                let roundedRect = RoundedRectangle(cornerRadius: cornerRadius)
                    .path(in: highlightRect)
                dimPath.addPath(roundedRect)

                context.fill(dimPath, with: .color(.black.opacity(0.45)), style: FillStyle(eoFill: true))

                // Glowing border around the cutout
                context.stroke(roundedRect, with: .color(.white.opacity(0.7)), lineWidth: 2.5)
            }
            .animation(.easeInOut(duration: 1.0).repeatForever(autoreverses: true), value: pulseAmount)
            .onAppear {
                pulseAmount = 1.0
            }
        }
        .ignoresSafeArea()
    }

    /// Convert normalized image-space bbox to screen coordinates.
    /// Accounts for portrait vs landscape buffer orientation.
    private func convertToScreen(bbox: CGRect, in screenSize: CGSize) -> CGRect {
        if bufferIsLandscape {
            // Buffer is landscape, screen is portrait:
            // image x → screen y (inverted), image y → screen x
            return CGRect(
                x: bbox.origin.y * screenSize.width,
                y: (1.0 - bbox.origin.x - bbox.width) * screenSize.height,
                width: bbox.height * screenSize.width,
                height: bbox.width * screenSize.height
            )
        } else {
            return CGRect(
                x: bbox.origin.x * screenSize.width,
                y: bbox.origin.y * screenSize.height,
                width: bbox.width * screenSize.width,
                height: bbox.height * screenSize.height
            )
        }
    }
}
