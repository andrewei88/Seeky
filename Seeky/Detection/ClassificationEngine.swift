import Foundation
import CoreImage

struct ClassificationResult {
    let word: String?        // nil when confidence too low but features available
    let confidence: Double   // raw softmax confidence (before threshold check)
    let secondConfidence: Double  // second-place softmax (for margin calculation)
    let features: [Float]?  // 1024-dim from custom model (for CorrectionStore)
}

final class ClassificationEngine {
    var customClassifier: CustomClassifier?
    private let correctionStore: CorrectionStore

    /// Minimum confidence to accept the custom classifier's answer.
    let acceptanceThreshold: Double

    /// Per-class confidence overrides for classes with high false-positive rates.
    /// Classes not listed here use the default acceptanceThreshold.
    static let perClassThresholds: [String: Double] = [
        "book": 0.75,       // high false-positive rate on non-book objects (cups at distance)
        "couch": 0.70,      // confused with bed/blanket/pillow, rarely reaches threshold anyway
        "cup": 0.30,        // cup is reliably top-1 at 0.30+ on real cups, never above 0.15 on non-cups
        "mushroom": 0.90,   // false positives on round textured surfaces (stuffed animals, cushions, rug patterns)
    ]

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
            return ClassificationResult(word: correctedWord, confidence: custom.confidence, secondConfidence: custom.secondConfidence, features: custom.features)
        }

        let w = CVPixelBufferGetWidth(imageBuffer)
        let h = CVPixelBufferGetHeight(imageBuffer)
        let margin = custom.secondConfidence > 0
            ? custom.confidence / custom.secondConfidence
            : Double.infinity

        print("[Classify] Buffer: \(w)x\(h), custom: '\(custom.word)' (conf=\(String(format: "%.3f", custom.confidence)), second=\(String(format: "%.3f", custom.secondConfidence)), margin=\(String(format: "%.1f", margin)))")

        let threshold = Self.perClassThresholds[custom.word] ?? acceptanceThreshold
        guard custom.confidence >= threshold else {
            print("[Classify] Rejected '\(custom.word)' (conf=\(String(format: "%.3f", custom.confidence)) < \(threshold)\(threshold != acceptanceThreshold ? " [per-class]" : ""))")
            return ClassificationResult(word: nil, confidence: custom.confidence, secondConfidence: custom.secondConfidence, features: custom.features)
        }

        print("[Classify] Accepted '\(custom.word)' (conf=\(String(format: "%.3f", custom.confidence)))")
        return ClassificationResult(word: custom.word, confidence: custom.confidence, secondConfidence: custom.secondConfidence, features: custom.features)
    }
}
