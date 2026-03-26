import Foundation
import CoreImage

struct ClassificationResult {
    let word: String?        // nil when confidence too low but features available
    let features: [Float]?  // 1024-dim from custom model (for CorrectionStore)
}

final class ClassificationEngine {
    var customClassifier: CustomClassifier?
    private let correctionStore: CorrectionStore

    /// Minimum confidence to accept the custom classifier's answer.
    let acceptanceThreshold: Double

    init(customClassifier: CustomClassifier? = nil, correctionStore: CorrectionStore, acceptanceThreshold: Double = 0.40) {
        self.customClassifier = customClassifier
        self.correctionStore = correctionStore
        self.acceptanceThreshold = acceptanceThreshold
    }

    /// Classify a cropped object image. Returns the child-friendly word or nil.
    func classify(imageBuffer: CVPixelBuffer) async -> ClassificationResult? {
        guard let custom = customClassifier?.classify(imageBuffer: imageBuffer) else {
            return nil
        }

        // Check stored corrections first (highest priority)
        if let correctedWord = await correctionStore.lookup(embedding: custom.features) {
            return ClassificationResult(word: correctedWord, features: custom.features)
        }

        let w = CVPixelBufferGetWidth(imageBuffer)
        let h = CVPixelBufferGetHeight(imageBuffer)
        let margin = custom.secondConfidence > 0
            ? custom.confidence / custom.secondConfidence
            : Double.infinity

        print("[Classify] Buffer: \(w)x\(h), custom: '\(custom.word)' (conf=\(String(format: "%.3f", custom.confidence)), second=\(String(format: "%.3f", custom.secondConfidence)), margin=\(String(format: "%.1f", margin)))")

        guard custom.confidence >= acceptanceThreshold else {
            print("[Classify] Rejected '\(custom.word)' (conf=\(String(format: "%.3f", custom.confidence)) < \(acceptanceThreshold))")
            return ClassificationResult(word: nil, features: custom.features)
        }

        print("[Classify] Accepted '\(custom.word)' (conf=\(String(format: "%.3f", custom.confidence)))")
        return ClassificationResult(word: custom.word, features: custom.features)
    }
}
