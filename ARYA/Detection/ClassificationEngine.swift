import Vision
import CoreImage

struct ClassificationResult {
    let word: String?        // nil when consensus gate rejects but features available
    let features: [Float]?  // 1024-dim from custom model (for CorrectionStore)
}

final class ClassificationEngine {
    private let labelMapper: LabelMapper
    private let customClassifier: CustomClassifier?
    private let consensusGate: ConsensusGate
    private let correctionStore: CorrectionStore

    init(labelMapper: LabelMapper, customClassifier: CustomClassifier?, correctionStore: CorrectionStore, consensusGate: ConsensusGate = ConsensusGate()) {
        self.labelMapper = labelMapper
        self.customClassifier = customClassifier
        self.correctionStore = correctionStore
        self.consensusGate = consensusGate
    }

    /// Classify a cropped object image. Returns the child-friendly word or nil.
    func classify(imageBuffer: CVPixelBuffer) async -> ClassificationResult? {
        async let vnResult = runVNClassify(imageBuffer)
        let customResult = customClassifier?.classify(imageBuffer: imageBuffer)

        let observations = await vnResult

        // Check stored corrections first (highest priority)
        if let features = customResult?.features,
           let correctedWord = await correctionStore.lookup(embedding: features) {
            return ClassificationResult(word: correctedWord, features: features)
        }

        // Log buffer dimensions for debugging crop issues
        let w = CVPixelBufferGetWidth(imageBuffer)
        let h = CVPixelBufferGetHeight(imageBuffer)
        print("[VNClassify] Buffer size: \(w)x\(h)")

        // Process VN observations
        var vnBest: (key: String, value: Float)?
        var vnSecondBest: (key: String, value: Float)?
        var wordConfidences: [String: Float] = [:]

        if let observations = observations, !observations.isEmpty {
            let top20 = observations.prefix(20).map { "\($0.identifier)(\(String(format: "%.3f", $0.confidence)))" }
            print("[VNClassify] Top 20: \(top20.joined(separator: ", "))")

            // Aggregate confidence across ALL VN labels that map to the same word.
            for obs in observations {
                if obs.confidence < 0.02 { break }

                let label = obs.identifier
                let mapped = labelMapper.childWord(for: label)
                    ?? labelMapper.childWord(for: label.replacingOccurrences(of: "_", with: " "))
                    ?? labelMapper.childWord(for: label.lowercased().replacingOccurrences(of: "_", with: " "))

                guard let word = mapped else { continue }
                wordConfidences[word, default: 0] += obs.confidence
            }

            let sorted = wordConfidences.sorted { $0.value > $1.value }
            vnBest = sorted.first
            vnSecondBest = sorted.count > 1 ? sorted[1] : nil

            if let best = vnBest {
                print("[Classify] VN aggregated: '\(best.key)' (conf=\(String(format: "%.4f", best.value))), " +
                      "second: '\(vnSecondBest?.key ?? "none")' (conf=\(String(format: "%.4f", vnSecondBest?.value ?? Float(0))))")
            } else {
                let allAboveThreshold = observations.filter { $0.confidence >= 0.02 }
                    .prefix(30)
                    .map { "\($0.identifier)(\(String(format: "%.3f", $0.confidence)))" }
                print("[Classify] No VN mapped word. All ≥0.02: \(allAboveThreshold.joined(separator: ", "))")
            }
        } else {
            print("[Classify] VNClassify returned nil")
        }

        // If custom classifier is available, use probability-based consensus
        if let custom = customResult {
            if let best = vnBest {
                let secondConfidence = vnSecondBest.map { Double($0.value) } ?? 0

                if let word = consensusGate.evaluate(
                    vnClassifyWord: best.key,
                    vnConfidence: Double(best.value),
                    vnSecondConfidence: secondConfidence,
                    customWord: custom.word,
                    customConfidence: custom.confidence,
                    customSecondConfidence: custom.secondConfidence
                ) {
                    return ClassificationResult(word: word, features: custom.features)
                }
                // Consensus rejected, but return features so user can correct
                return ClassificationResult(word: nil, features: custom.features)
            }

            // VN has NO mapped word — trust custom classifier alone at lower bar
            guard custom.confidence >= consensusGate.customOnlyConfidence else {
                print("[Classify] Custom-only: rejected '\(custom.word)' (conf=\(String(format: "%.3f", custom.confidence)) < \(consensusGate.customOnlyConfidence))")
                // Return features so user can correct
                return ClassificationResult(word: nil, features: custom.features)
            }

            print("[Classify] Custom-only (VN had no mapped word): '\(custom.word)' (conf=\(String(format: "%.3f", custom.confidence)))")
            return ClassificationResult(word: custom.word, features: custom.features)
        }

        // No custom classifier available — VN-only with permissive thresholds
        guard let best = vnBest else { return nil }

        let margin = vnSecondBest != nil ? best.value / vnSecondBest!.value : Float.infinity
        let minConfidence: Float = 0.03

        guard best.value >= minConfidence else {
            print("[Classify] VN-only: rejected '\(best.key)' (conf=\(best.value) < \(minConfidence))")
            return nil
        }

        if let second = vnSecondBest, margin < 1.05 {
            print("[Classify] VN-only: rejected '\(best.key)' — ambiguous with '\(second.key)' (margin=\(String(format: "%.2f", margin)))")
            return nil
        }

        print("[Classify] VN-only: accepted '\(best.key)' (conf=\(String(format: "%.4f", best.value)), margin=\(String(format: "%.2f", margin)))")
        return ClassificationResult(word: best.key, features: nil)
    }

    private func runVNClassify(_ buffer: CVPixelBuffer) async -> [VNClassificationObservation]? {
        await withCheckedContinuation { continuation in
            let request = VNClassifyImageRequest { request, error in
                guard let observations = request.results as? [VNClassificationObservation],
                      !observations.isEmpty else {
                    continuation.resume(returning: nil)
                    return
                }

                continuation.resume(returning: observations)
            }

            let handler = VNImageRequestHandler(cvPixelBuffer: buffer, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("[VNClassify] Error: \(error)")
                continuation.resume(returning: nil)
            }
        }
    }
}
