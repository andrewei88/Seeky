import SwiftUI
import UIKit

/// Draws a dimmed overlay with a rounded-rect cutout around the detected object,
/// plus a prominent gold glow border. Uses CAShapeLayer for GPU-accelerated rendering.
/// With VNTrackObjectRequest providing 30fps updates, animation duration is kept short.
struct ObjectHighlightView: UIViewRepresentable {
    let boundingBox: CGRect   // normalized 0-1 in image coordinates
    let bufferIsLandscape: Bool

    func makeUIView(context: Context) -> HighlightUIView {
        HighlightUIView()
    }

    func updateUIView(_ uiView: HighlightUIView, context: Context) {
        uiView.updateHighlight(normalizedRect: boundingBox, bufferIsLandscape: bufferIsLandscape)
    }
}

class HighlightUIView: UIView {
    private let dimLayer = CAShapeLayer()
    private let outerGlowLayer = CAShapeLayer()
    private let glowLayer = CAShapeLayer()
    private let cornerRadius: CGFloat = 20

    override init(frame: CGRect) {
        super.init(frame: frame)
        isUserInteractionEnabled = false
        backgroundColor = .clear
        clipsToBounds = false
        layer.masksToBounds = false

        // Dim layer: black with cutout
        dimLayer.fillRule = .evenOdd
        dimLayer.fillColor = UIColor.black.withAlphaComponent(0.45).cgColor
        layer.addSublayer(dimLayer)

        let gold = UIColor(red: 1.0, green: 0.84, blue: 0.0, alpha: 1.0)

        // Outer glow: wide, soft halo
        outerGlowLayer.fillColor = UIColor.clear.cgColor
        outerGlowLayer.strokeColor = gold.withAlphaComponent(0.4).cgColor
        outerGlowLayer.lineWidth = 12
        outerGlowLayer.shadowColor = gold.cgColor
        outerGlowLayer.shadowOpacity = 0.9
        outerGlowLayer.shadowRadius = 40
        outerGlowLayer.shadowOffset = .zero
        layer.addSublayer(outerGlowLayer)

        // Inner glow: bright core stroke
        glowLayer.fillColor = UIColor.clear.cgColor
        glowLayer.strokeColor = gold.cgColor
        glowLayer.lineWidth = 4
        glowLayer.shadowColor = gold.cgColor
        glowLayer.shadowOpacity = 1.0
        glowLayer.shadowRadius = 15
        glowLayer.shadowOffset = .zero
        layer.addSublayer(glowLayer)

        // Subtle pulse animation
        let pulse = CABasicAnimation(keyPath: "shadowRadius")
        pulse.fromValue = 12
        pulse.toValue = 22
        pulse.duration = 1.2
        pulse.autoreverses = true
        pulse.repeatCount = .infinity
        pulse.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        glowLayer.add(pulse, forKey: "pulse")

        let outerPulse = CABasicAnimation(keyPath: "shadowRadius")
        outerPulse.fromValue = 30
        outerPulse.toValue = 50
        outerPulse.duration = 1.2
        outerPulse.autoreverses = true
        outerPulse.repeatCount = .infinity
        outerPulse.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        outerGlowLayer.add(outerPulse, forKey: "pulse")
    }

    required init?(coder: NSCoder) { fatalError() }

    override func layoutSubviews() {
        super.layoutSubviews()
        dimLayer.frame = bounds
        outerGlowLayer.frame = bounds
        glowLayer.frame = bounds
    }

    func updateHighlight(normalizedRect: CGRect, bufferIsLandscape: Bool) {
        guard bounds.width > 0, bounds.height > 0 else { return }
        guard normalizedRect.width > 0, normalizedRect.height > 0 else { return }

        let screenRect = mapToScreen(normalizedRect, bufferIsLandscape: bufferIsLandscape)

        // Dim: full screen path with rounded-rect cutout (even-odd fill)
        let fullPath = UIBezierPath(rect: bounds)
        let cutout = UIBezierPath(roundedRect: screenRect, cornerRadius: cornerRadius)
        fullPath.append(cutout)

        // Glow paths: same rounded rect
        let glowPath = UIBezierPath(roundedRect: screenRect, cornerRadius: cornerRadius)

        // Short animation for smooth 30fps tracking interpolation
        let animDuration: CFTimeInterval = 0.05
        let timing = CAMediaTimingFunction(name: .easeOut)

        if dimLayer.path != nil {
            for targetLayer in [dimLayer, outerGlowLayer, glowLayer] {
                let anim = CABasicAnimation(keyPath: "path")
                anim.duration = animDuration
                anim.timingFunction = timing
                targetLayer.add(anim, forKey: "path")
            }
        }

        dimLayer.path = fullPath.cgPath
        outerGlowLayer.path = glowPath.cgPath
        glowLayer.path = glowPath.cgPath
    }

    /// Maps normalized image coordinates to screen coordinates, accounting for aspect-fill.
    private func mapToScreen(_ rect: CGRect, bufferIsLandscape: Bool) -> CGRect {
        let screenW = bounds.width
        let screenH = bounds.height

        // Rotate bounding box from image coordinates to screen coordinates if needed
        let rotated: CGRect
        if bufferIsLandscape {
            rotated = CGRect(
                x: 1.0 - rect.origin.y - rect.height,
                y: rect.origin.x,
                width: rect.height,
                height: rect.width
            )
        } else {
            rotated = rect
        }

        // Aspect-fill mapping: camera 9:16 → screen aspect ratio
        let imageAspect: CGFloat = 9.0 / 16.0
        let screenAspect = screenW / screenH

        let scale: CGFloat
        let offsetX: CGFloat
        let offsetY: CGFloat

        if screenAspect > imageAspect {
            scale = screenW / imageAspect / screenH
            offsetX = 0
            offsetY = (1.0 - 1.0 / scale) / 2.0
        } else {
            scale = screenH * imageAspect / screenW
            offsetX = (1.0 - 1.0 / scale) / 2.0
            offsetY = 0
        }

        let mappedX = (rotated.origin.x - offsetX) * scale * screenW
        let mappedY = (rotated.origin.y - offsetY) * scale * screenH
        let mappedW = rotated.width * scale * screenW
        let mappedH = rotated.height * scale * screenH

        return CGRect(x: mappedX, y: mappedY, width: mappedW, height: mappedH)
    }
}
