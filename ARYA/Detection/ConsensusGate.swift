import Foundation

struct ConsensusGate {
    let clipSimilarityThreshold: Double
    let clipMarginMultiplier: Double

    init(
        clipSimilarityThreshold: Double = 0.20,
        clipMarginMultiplier: Double = 1.05
    ) {
        self.clipSimilarityThreshold = clipSimilarityThreshold
        self.clipMarginMultiplier = clipMarginMultiplier
    }

    /// Evaluates dual-model consensus between VN and CLIP.
    /// Strategy:
    /// - If both agree → accept (highest confidence)
    /// - If they disagree → trust CLIP if it meets similarity threshold
    /// - VN is used as a tiebreaker / confirmation signal
    func evaluate(
        vnClassifyWord: String,
        vnConfidence: Double,
        vnSecondConfidence: Double,
        clipWord: String,
        clipSimilarity: Double,
        clipSecondSimilarity: Double
    ) -> String? {
        let clipMargin = clipSecondSimilarity > 0
            ? clipSimilarity / clipSecondSimilarity
            : Double.infinity

        // Both models agree — accept with lower bar
        if vnClassifyWord == clipWord {
            guard clipSimilarity >= clipSimilarityThreshold else {
                print("[Consensus] Both agree on '\(clipWord)' but CLIP similarity too low (\(String(format: "%.3f", clipSimilarity)))")
                return nil
            }
            print("[Consensus] Agreement: '\(clipWord)' (CLIP=\(String(format: "%.3f", clipSimilarity)), VN=\(String(format: "%.4f", vnConfidence)))")
            return clipWord
        }

        // Models disagree — trust CLIP if it's confident and has margin
        guard clipSimilarity >= clipSimilarityThreshold else {
            print("[Consensus] Disagree: VN='\(vnClassifyWord)', CLIP='\(clipWord)' — CLIP too low (\(String(format: "%.3f", clipSimilarity)))")
            return nil
        }

        guard clipMargin >= clipMarginMultiplier else {
            print("[Consensus] Disagree: VN='\(vnClassifyWord)', CLIP='\(clipWord)' — CLIP margin too narrow (\(String(format: "%.2f", clipMargin)))")
            return nil
        }

        print("[Consensus] CLIP overrides VN: '\(clipWord)' (CLIP=\(String(format: "%.3f", clipSimilarity)), margin=\(String(format: "%.2f", clipMargin))) vs VN='\(vnClassifyWord)'")
        return clipWord
    }
}
