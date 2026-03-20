import SwiftUI
import Vision

enum AppMode: Equatable {
    case exploring
    case classifying // Brief transition while classification runs
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
    var instanceTracker = InstanceTracker()

    // Latest segmentation result for mask access
    @Published var latestSegmentation: SegmentationResult?

    init() {
        hasCompletedFirstTap = UserDefaults.standard.bool(forKey: "hasCompletedFirstTap")
        vocabularyStore = VocabularyStore.load()
        labelMapper = LabelMapper.load()

        let clipEmbeddings = CLIPEmbeddings.load(vocabulary: vocabularyStore.entries.map(\.word))
        classificationEngine = ClassificationEngine(labelMapper: labelMapper, clipEmbeddings: clipEmbeddings)

        cameraManager.delegate = self
    }

    func handleTap(at normalizedPoint: CGPoint) {
        guard mode == .exploring else { return }

        // Check if tap hit a tappable instance
        guard let instance = instanceTracker.instance(at: normalizedPoint) else { return }

        // Trigger haptic
        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()

        mode = .classifying

        guard let segResult = latestSegmentation else {
            mode = .exploring
            return
        }

        Task {
            // Crop the instance region from the full frame
            let croppedBuffer = cropPixelBuffer(segResult.pixelBuffer, to: instance.boundingBox)

            guard let croppedBuffer = croppedBuffer,
                  let result = await classificationEngine.classify(imageBuffer: croppedBuffer) else {
                mode = .exploring
                return
            }

            // Mark first tap complete
            if !hasCompletedFirstTap {
                hasCompletedFirstTap = true
                UserDefaults.standard.set(true, forKey: "hasCompletedFirstTap")
            }

            mode = .learning(word: result.word, instanceIndex: instance.id)
        }
    }

    func dismissLearning() {
        wordSpeaker.stop()
        mode = .exploring
    }

    private func cropPixelBuffer(_ buffer: CVPixelBuffer, to normalizedRect: CGRect) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)

        let cropRect = CGRect(
            x: normalizedRect.origin.x * CGFloat(width),
            y: normalizedRect.origin.y * CGFloat(height),
            width: normalizedRect.width * CGFloat(width),
            height: normalizedRect.height * CGFloat(height)
        ).integral

        guard cropRect.width > 0, cropRect.height > 0 else { return nil }

        let ciImage = CIImage(cvPixelBuffer: buffer).cropped(to: cropRect)
        let context = CIContext()
        var croppedBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(cropRect.width), Int(cropRect.height),
                           kCVPixelFormatType_32BGRA, nil, &croppedBuffer)
        guard let output = croppedBuffer else { return nil }
        context.render(ciImage, to: output)
        return output
    }
}

extension AppState: CameraManagerDelegate {
    nonisolated func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        // Run segmentation off main thread, then update state on main
        let result = segmentationEngine.segment(pixelBuffer: pixelBuffer)
        let timeSeconds = CMTimeGetSeconds(timestamp)

        Task { @MainActor in
            instanceTracker.update(instances: result.instances, timestamp: timeSeconds)
            latestSegmentation = result
        }
    }
}
