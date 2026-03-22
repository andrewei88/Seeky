import SwiftUI
import Vision

enum AppMode: Equatable {
    case exploring
    case classifying
    case learning(word: String, instanceIndex: Int)
}

@MainActor
final class AppState: ObservableObject {
    @Published var mode: AppMode = .exploring
    @Published private(set) var hasCompletedFirstTap: Bool

    let cameraManager = CameraManager()
    let segmentationEngine = SegmentationEngine()
    let wordSpeaker = WordSpeaker()
    let vocabularyStore: VocabularyStore
    let labelMapper: LabelMapper
    let classificationEngine: ClassificationEngine
    var instanceTracker = InstanceTracker(tapPadding: 0.08)
    let objectTracker = ObjectTracker()

    // Live segmentation (updates every frame during exploring)
    private var liveSegmentation: SegmentationResult?

    // Bounding box of tracked object (updated at 30fps by ObjectTracker)
    @Published var learningBoundingBox: CGRect = .zero

    // Raw screen-space tap location (points) for glow effect during learning
    @Published var tapScreenPoint: CGPoint = .zero

    init() {
        hasCompletedFirstTap = UserDefaults.standard.bool(forKey: "hasCompletedFirstTap")
        vocabularyStore = VocabularyStore.load()
        labelMapper = LabelMapper.load()

        let clipEmbeddings = CLIPEmbeddings.load(vocabulary: vocabularyStore.entries.map(\.word))
        classificationEngine = ClassificationEngine(labelMapper: labelMapper, clipEmbeddings: clipEmbeddings)

        cameraManager.delegate = self
    }

    /// Whether camera buffers are landscape (needs coordinate transform) or portrait (direct mapping).
    /// Set from the first camera frame's dimensions.
    var bufferIsLandscape: Bool = false

    func handleTap(imagePoint: CGPoint, screenPoint: CGPoint) {
        guard mode == .exploring else { return }

        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()

        tapScreenPoint = screenPoint
        mode = .classifying

        guard let segResult = liveSegmentation else {
            print("[Tap] No segmentation data available")
            mode = .exploring
            return
        }

        let bufW = CVPixelBufferGetWidth(segResult.pixelBuffer)
        let bufH = CVPixelBufferGetHeight(segResult.pixelBuffer)

        // captureDevicePointConverted returns coordinates in the capture device's
        // native sensor space (landscape). The pixel buffer is rotated 90° CW to portrait
        // via videoRotationAngle=90. Convert: buffer(x,y) = (1 - device.y, device.x)
        let bufferPoint = CGPoint(x: 1.0 - imagePoint.y, y: imagePoint.x)

        let cropRect = tapCenteredCropRect(tapPoint: bufferPoint, cropFraction: 0.25,
                                           bufferWidth: bufW, bufferHeight: bufH)
        print("[Tap] screen=\(screenPoint) → device=\(imagePoint) → buffer=\(bufferPoint), crop=\(cropRect)")

        Task {
            let croppedBuffer = cropPixelBuffer(segResult.pixelBuffer, to: cropRect)

            guard let croppedBuffer = croppedBuffer else {
                print("[Tap] Failed to crop pixel buffer")
                mode = .exploring
                return
            }

            guard let result = await classificationEngine.classify(imageBuffer: croppedBuffer) else {
                print("[Tap] Classification returned nil (thresholds not met)")
                mode = .exploring
                return
            }

            if !hasCompletedFirstTap {
                hasCompletedFirstTap = true
                UserDefaults.standard.set(true, forKey: "hasCompletedFirstTap")
            }

            mode = .learning(word: result.word, instanceIndex: 0)
        }
    }

    func dismissLearning() {
        wordSpeaker.stop()
        mode = .exploring
    }

    /// Computes a normalized crop rect centered on the tap point.
    /// Uses cropFraction of the smaller frame dimension as the crop size.
    private func tapCenteredCropRect(tapPoint: CGPoint, cropFraction: CGFloat, bufferWidth: Int, bufferHeight: Int) -> CGRect {
        let w = CGFloat(bufferWidth)
        let h = CGFloat(bufferHeight)
        let minDim = min(w, h)
        let cropPixels = minDim * cropFraction

        // Normalized crop size
        let cropW = cropPixels / w
        let cropH = cropPixels / h

        var x = tapPoint.x - cropW / 2
        var y = tapPoint.y - cropH / 2

        // Clamp to [0, 1] bounds
        x = max(0, min(x, 1.0 - cropW))
        y = max(0, min(y, 1.0 - cropH))

        return CGRect(x: x, y: y, width: cropW, height: cropH)
    }

    private func cropPixelBuffer(_ buffer: CVPixelBuffer, to normalizedRect: CGRect) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)

        // CIImage uses bottom-left origin. normalizedRect uses top-left origin.
        // Flip Y: ciY = bufferHeight - topLeftY - cropHeight
        let pxX = normalizedRect.origin.x * CGFloat(width)
        let pxW = normalizedRect.width * CGFloat(width)
        let pxH = normalizedRect.height * CGFloat(height)
        let pxY = CGFloat(height) - normalizedRect.origin.y * CGFloat(height) - pxH

        let cropRect = CGRect(x: pxX, y: pxY, width: pxW, height: pxH).integral

        guard cropRect.width > 0, cropRect.height > 0 else { return nil }

        // Crop, then translate extent to (0,0) so CIContext.render intersects the output buffer
        let cropped = CIImage(cvPixelBuffer: buffer)
            .cropped(to: cropRect)
            .transformed(by: CGAffineTransform(translationX: -cropRect.origin.x, y: -cropRect.origin.y))

        let context = CIContext()
        var croppedBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(cropRect.width), Int(cropRect.height),
                           kCVPixelFormatType_32BGRA, nil, &croppedBuffer)
        guard let output = croppedBuffer else { return nil }
        context.render(cropped, to: output)
        return output
    }

}

extension AppState: CameraManagerDelegate {
    nonisolated func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        let bufW = CVPixelBufferGetWidth(pixelBuffer)
        let bufH = CVPixelBufferGetHeight(pixelBuffer)
        let timeSeconds = CMTimeGetSeconds(timestamp)

        struct Once { static var logged = false }
        if !Once.logged {
            let isLandscape = bufW > bufH
            print("[Camera] Buffer dimensions: \(bufW)×\(bufH) (\(isLandscape ? "LANDSCAPE" : "PORTRAIT"))")
            Once.logged = true
            Task { @MainActor [isLandscape] in
                self.bufferIsLandscape = isLandscape
            }
        }

        // Exploring mode: run segmentation (~3fps)
        let result = segmentationEngine.segment(pixelBuffer: pixelBuffer)

        if !result.instances.isEmpty {
            print("[Seg] Found \(result.instances.count) instances at t=\(String(format: "%.1f", timeSeconds))")
        }

        Task { @MainActor in
            if mode == .exploring {
                instanceTracker.update(instances: result.instances, timestamp: timeSeconds)
                liveSegmentation = result
            }
        }
    }
}
