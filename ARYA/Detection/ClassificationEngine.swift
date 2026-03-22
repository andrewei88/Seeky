import Vision
import CoreImage
import UIKit

struct ClassificationResult {
    let word: String
    let clipEmbedding: [Float]?
}

final class ClassificationEngine {
    private let labelMapper: LabelMapper
    private let clipEmbeddings: CLIPEmbeddings?
    private let consensusGate: ConsensusGate
    var correctionStore: CorrectionStore?

    init(labelMapper: LabelMapper, clipEmbeddings: CLIPEmbeddings?, consensusGate: ConsensusGate = ConsensusGate()) {
        self.labelMapper = labelMapper
        self.clipEmbeddings = clipEmbeddings
        self.consensusGate = consensusGate
    }

    /// Classify a cropped object image. Returns the child-friendly word or nil.
    func classify(imageBuffer: CVPixelBuffer) async -> ClassificationResult? {
        async let vnResult = runVNClassify(imageBuffer)
        async let clipResult = runCLIP(imageBuffer)

        let observations = await vnResult
        let clip = await clipResult

        // Check stored corrections first (highest priority)
        if let embedding = clip?.embedding, let correctionStore = correctionStore,
           let correctedWord = correctionStore.lookup(embedding: embedding) {
            return ClassificationResult(word: correctedWord, clipEmbedding: embedding)
        }

        // Log buffer dimensions for debugging crop issues
        let w = CVPixelBufferGetWidth(imageBuffer)
        let h = CVPixelBufferGetHeight(imageBuffer)
        print("[VNClassify] Buffer size: \(w)x\(h)")

        // Process VN observations
        var vnBest: (key: String, value: Float)?
        var vnSecondBest: (key: String, value: Float)?

        if let observations = observations, !observations.isEmpty {
            // Log top 20 for debugging (shows what VN actually sees)
            let top20 = observations.prefix(20).map { "\($0.identifier)(\(String(format: "%.3f", $0.confidence)))" }
            print("[VNClassify] Top 20: \(top20.joined(separator: ", "))")

            // Aggregate confidence across ALL VN labels that map to the same word.
            var wordConfidences: [String: Float] = [:]

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

        // If CLIP is available, use dual-model logic
        if let clip = clip {
            // If VN has a mapped word, use consensus gate
            if let best = vnBest {
                let secondConfidence = vnSecondBest.map { Double($0.value) } ?? 0

                if let word = consensusGate.evaluate(
                    vnClassifyWord: best.key,
                    vnConfidence: Double(best.value),
                    vnSecondConfidence: secondConfidence,
                    clipWord: clip.top1.word,
                    clipSimilarity: clip.top1.similarity,
                    clipSecondSimilarity: clip.top2.similarity
                ) {
                    return ClassificationResult(word: word, clipEmbedding: clip.embedding)
                }
                return nil
            }

            // VN has NO mapped word (e.g., document/screenshot) — trust CLIP alone
            let clipMargin = clip.top2.similarity > 0
                ? clip.top1.similarity / clip.top2.similarity
                : Double.infinity

            guard clip.top1.similarity >= consensusGate.clipSimilarityThreshold else {
                print("[Classify] CLIP-only: rejected '\(clip.top1.word)' (sim=\(String(format: "%.3f", clip.top1.similarity)) < threshold)")
                return nil
            }

            guard clipMargin >= consensusGate.clipMarginMultiplier else {
                print("[Classify] CLIP-only: rejected '\(clip.top1.word)' — ambiguous (margin=\(String(format: "%.2f", clipMargin)))")
                return nil
            }

            print("[Classify] CLIP-only (VN had no mapped word): '\(clip.top1.word)' (sim=\(String(format: "%.3f", clip.top1.similarity)), margin=\(String(format: "%.2f", clipMargin)))")
            return ClassificationResult(word: clip.top1.word, clipEmbedding: clip.embedding)
        }

        // No CLIP available — VN-only with permissive thresholds
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
        return ClassificationResult(word: best.key, clipEmbedding: nil)
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
                continuation.resume(returning: nil)
            }
        }
    }

    private func runCLIP(_ buffer: CVPixelBuffer) async -> (top1: CLIPResult, top2: CLIPResult, embedding: [Float])? {
        clipEmbeddings?.classify(imageBuffer: buffer)
    }
}
