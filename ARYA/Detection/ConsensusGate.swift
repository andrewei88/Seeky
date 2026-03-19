import Foundation

struct ConsensusGate {
    let vnConfidenceThreshold: Double
    let vnMarginMultiplier: Double
    let clipSimilarityThreshold: Double
    let clipMarginMultiplier: Double

    init(
        vnConfidenceThreshold: Double = 0.70,
        vnMarginMultiplier: Double = 1.5,
        clipSimilarityThreshold: Double = 0.75,
        clipMarginMultiplier: Double = 1.3
    ) {
        self.vnConfidenceThreshold = vnConfidenceThreshold
        self.vnMarginMultiplier = vnMarginMultiplier
        self.clipSimilarityThreshold = clipSimilarityThreshold
        self.clipMarginMultiplier = clipMarginMultiplier
    }

    /// Returns the agreed-upon child word if both classifiers agree with sufficient confidence, nil otherwise.
    func evaluate(
        vnClassifyWord: String?,
        vnConfidence: Double,
        vnSecondConfidence: Double,
        clipWord: String,
        clipSimilarity: Double,
        clipSecondSimilarity: Double
    ) -> String? {
        // VN must have a mapped word
        guard let vnWord = vnClassifyWord else { return nil }

        // Both must agree
        guard vnWord == clipWord else { return nil }

        // VN confidence gate
        guard vnConfidence >= vnConfidenceThreshold else { return nil }

        // VN margin gate
        guard vnSecondConfidence == 0 || vnConfidence >= vnSecondConfidence * vnMarginMultiplier else { return nil }

        // CLIP similarity gate
        guard clipSimilarity >= clipSimilarityThreshold else { return nil }

        // CLIP margin gate
        guard clipSecondSimilarity == 0 || clipSimilarity >= clipSecondSimilarity * clipMarginMultiplier else { return nil }

        return vnWord
    }
}
