import SwiftUI
import Vision
import CoreImage

struct MaskOverlayView: UIViewRepresentable {
    let observation: VNInstanceMaskObservation?
    let instanceIndex: Int
    let pixelBuffer: CVPixelBuffer?

    func makeUIView(context: Context) -> MaskUIView {
        MaskUIView()
    }

    func updateUIView(_ uiView: MaskUIView, context: Context) {
        uiView.updateMask(observation: observation, instanceIndex: instanceIndex, pixelBuffer: pixelBuffer)
    }
}

class MaskUIView: UIView {
    private let dimLayer = CALayer()
    private let glowLayer = CALayer()
    private let ciContext = CIContext()

    override init(frame: CGRect) {
        super.init(frame: frame)
        isUserInteractionEnabled = false
        backgroundColor = .clear
        layer.addSublayer(dimLayer)
        layer.addSublayer(glowLayer)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func layoutSubviews() {
        super.layoutSubviews()
        dimLayer.frame = bounds
        glowLayer.frame = bounds
    }

    func updateMask(observation: VNInstanceMaskObservation?, instanceIndex: Int, pixelBuffer: CVPixelBuffer?) {
        guard let observation = observation, let pixelBuffer = pixelBuffer else {
            dimLayer.contents = nil
            glowLayer.contents = nil
            return
        }

        guard let maskBuffer = try? observation.generateScaledMaskForImage(
            forInstances: IndexSet(integer: instanceIndex),
            from: VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        ) else { return }

        let maskCI = CIImage(cvPixelBuffer: maskBuffer)

        // Dim layer: invert mask (everything EXCEPT the object is dimmed)
        let invertedMask = maskCI.applyingFilter("CIColorInvert")
        let dimImage = invertedMask.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0.6),
        ])

        if let cgDim = ciContext.createCGImage(dimImage, from: dimImage.extent) {
            dimLayer.contents = cgDim
            dimLayer.contentsGravity = .resizeAspectFill
        }

        // Glow layer: blur the mask edges for a gold glow effect
        let goldMask = maskCI.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 1.0, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: 0.85, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0.24, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0.8),
        ])
        let blurredGlow = goldMask.applyingFilter("CIGaussianBlur", parameters: ["inputRadius": 12.0])

        if let cgGlow = ciContext.createCGImage(blurredGlow, from: blurredGlow.extent) {
            glowLayer.contents = cgGlow
            glowLayer.contentsGravity = .resizeAspectFill
        }
    }
}
