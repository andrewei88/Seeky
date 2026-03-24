import Foundation

struct ConsensusGate {
    /// Custom classifier thresholds (softmax probabilities 0-1)
    let customHighConfidence: Double     // >this → accept custom without VN (very high bar)
    let customMediumConfidence: Double   // >this → check VN agreement or VN trust
    let customLowConfidence: Double      // <this → VN fallback only
    let customOnlyConfidence: Double     // >this → accept custom when VN has no mapped word
    let vnTrustThreshold: Double         // when VN >this and disagrees, trust VN over custom

    init(
        customHighConfidence: Double = 0.90,
        customMediumConfidence: Double = 0.40,
        customLowConfidence: Double = 0.10,
        customOnlyConfidence: Double = 0.45,
        vnTrustThreshold: Double = 0.45
    ) {
        self.customHighConfidence = customHighConfidence
        self.customMediumConfidence = customMediumConfidence
        self.customLowConfidence = customLowConfidence
        self.customOnlyConfidence = customOnlyConfidence
        self.vnTrustThreshold = vnTrustThreshold
    }

    /// Evaluates dual-model consensus between VN and custom classifier.
    ///
    /// Decision logic:
    /// - Custom >0.90 → accept custom (very high confidence)
    /// - Custom 0.40-0.90, VN agrees → accept custom word
    /// - Custom >=0.70, margin >=5.0, VN <0.60 → trust custom (overrides marginal VN)
    /// - Custom 0.40-0.90, VN disagrees + VN >0.45 → trust VN
    /// - Custom 0.40-0.90, VN weak + custom margin strong → trust custom
    /// - Custom <0.10 → fall back to VN if VN is confident
    func evaluate(
        vnClassifyWord: String,
        vnConfidence: Double,
        vnSecondConfidence: Double,
        customWord: String,
        customConfidence: Double,
        customSecondConfidence: Double
    ) -> String? {
        let customMargin = customSecondConfidence > 0
            ? customConfidence / customSecondConfidence
            : Double.infinity

        // Very high confidence from custom classifier — trust it
        // UNLESS VN very strongly disagrees (>= 0.80), which suggests
        // custom is outside its training distribution for this input.
        if customConfidence >= customHighConfidence {
            if vnClassifyWord != customWord && vnConfidence >= 0.80 {
                print("[Consensus] Custom high-conf BUT VN strongly disagrees: custom='\(customWord)'(\(String(format: "%.3f", customConfidence))), VN='\(vnClassifyWord)'(\(String(format: "%.4f", vnConfidence))) — trusting VN")
                return vnClassifyWord
            }
            print("[Consensus] Custom high-conf: '\(customWord)' (conf=\(String(format: "%.3f", customConfidence)), margin=\(String(format: "%.1f", customMargin)))")
            return customWord
        }

        // Medium confidence — check agreement or defer to strong VN
        if customConfidence >= customMediumConfidence {
            if vnClassifyWord == customWord {
                print("[Consensus] Agreement at medium conf: '\(customWord)' (custom=\(String(format: "%.3f", customConfidence)), VN=\(String(format: "%.4f", vnConfidence)))")
                return customWord
            }

            // Very strong custom overrides marginally-confident VN.
            // When custom is overwhelmingly confident with a dominant margin, VN
            // barely clearing its trust threshold shouldn't override that signal.
            if customConfidence >= 0.70 && customMargin >= 5.0 && vnConfidence < 0.60 {
                print("[Consensus] Custom override (very strong custom, marginal VN): '\(customWord)' (custom=\(String(format: "%.3f", customConfidence)), margin=\(String(format: "%.1f", customMargin)), VN='\(vnClassifyWord)'(\(String(format: "%.4f", vnConfidence))))")
                return customWord
            }

            // Disagreement — if VN is strongly confident, trust VN
            if vnConfidence >= vnTrustThreshold {
                print("[Consensus] VN trusted (strong VN, disagreement): '\(vnClassifyWord)' (VN=\(String(format: "%.4f", vnConfidence)), custom='\(customWord)'(\(String(format: "%.3f", customConfidence))))")
                return vnClassifyWord
            }

            // VN is below trust threshold — custom has decent margin? Trust custom.
            // VN often returns moderate confidence (0.20-0.45) on unrelated labels,
            // so we don't require VN to be near-zero for custom to win.
            if customMargin >= 2.0 {
                print("[Consensus] Custom override (strong margin, untrusted VN): '\(customWord)' (custom=\(String(format: "%.3f", customConfidence)), margin=\(String(format: "%.1f", customMargin)), VN=\(String(format: "%.4f", vnConfidence)))")
                return customWord
            }

            print("[Consensus] Disagree at medium conf: custom='\(customWord)'(\(String(format: "%.3f", customConfidence))), VN='\(vnClassifyWord)'(\(String(format: "%.4f", vnConfidence))) — rejected")
            return nil
        }

        // Low confidence from custom — fall back to VN
        if customConfidence < customLowConfidence {
            let vnMargin = vnSecondConfidence > 0 ? vnConfidence / vnSecondConfidence : Double.infinity
            if vnConfidence >= 0.05 && vnMargin >= 1.3 {
                print("[Consensus] VN fallback: '\(vnClassifyWord)' (VN=\(String(format: "%.4f", vnConfidence)), margin=\(String(format: "%.1f", vnMargin)))")
                return vnClassifyWord
            }
            print("[Consensus] Both models low confidence — rejected")
            return nil
        }

        // Between low and medium (0.10-0.40) — need agreement or strong VN
        if vnClassifyWord == customWord && vnConfidence >= 0.05 {
            print("[Consensus] Weak agreement: '\(customWord)' (custom=\(String(format: "%.3f", customConfidence)), VN=\(String(format: "%.4f", vnConfidence)))")
            return customWord
        }

        // VN is strong even though custom is weak — trust VN
        if vnConfidence >= vnTrustThreshold {
            print("[Consensus] VN trusted (weak custom): '\(vnClassifyWord)' (VN=\(String(format: "%.4f", vnConfidence)), custom='\(customWord)'(\(String(format: "%.3f", customConfidence))))")
            return vnClassifyWord
        }

        print("[Consensus] Custom='\(customWord)'(\(String(format: "%.3f", customConfidence))), VN='\(vnClassifyWord)'(\(String(format: "%.4f", vnConfidence))) — insufficient confidence")
        return nil
    }
}
