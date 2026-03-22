import Vision
import CoreImage
import UIKit

struct ClassificationResult {
    let word: String
}

final class ClassificationEngine {
    private let labelMapper: LabelMapper
    private let clipEmbeddings: CLIPEmbeddings?
    private let consensusGate: ConsensusGate

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

        guard let observations = observations, !observations.isEmpty else {
            print("[Classify] VNClassify returned nil")
            return nil
        }

        // Log top 20 for debugging (shows what VN actually sees)
        let top20 = observations.prefix(20).map { "\($0.identifier)(\(String(format: "%.3f", $0.confidence)))" }
        print("[VNClassify] Top 20: \(top20.joined(separator: ", "))")

        // Log buffer dimensions for debugging crop issues
        let w = CVPixelBufferGetWidth(imageBuffer)
        let h = CVPixelBufferGetHeight(imageBuffer)
        print("[VNClassify] Buffer size: \(w)x\(h)")

        // Aggregate confidence across ALL VN labels that map to the same word.
        // This lets e.g. light(0.04)+light_bulb(0.04)+spotlight(0.02) compete with moon(0.12).
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

        // Sort by aggregated confidence
        let sorted = wordConfidences.sorted { $0.value > $1.value }

        guard let best = sorted.first else {
            let allAboveThreshold = observations.filter { $0.confidence >= 0.02 }
                .prefix(30)
                .map { "\($0.identifier)(\(String(format: "%.3f", $0.confidence)))" }
            print("[Classify] No mapped word found. All VN results ≥0.02: \(allAboveThreshold.joined(separator: ", "))")
            return nil
        }

        let secondBest = sorted.count > 1 ? sorted[1] : nil

        print("[Classify] Aggregated: '\(best.key)' (conf=\(String(format: "%.4f", best.value))), " +
              "second: '\(secondBest?.key ?? "none")' (conf=\(String(format: "%.4f", secondBest?.value ?? 0)))")

        // If no CLIP available, use VN alone with permissive thresholds
        guard let clip = clip else {
            let margin = secondBest != nil ? best.value / secondBest!.value : Float.infinity
            let minConfidence: Float = 0.03

            guard best.value >= minConfidence else {
                print("[Classify] VN-only: rejected '\(best.key)' (conf=\(best.value) < \(minConfidence))")
                return nil
            }

            // Reject if two different mapped words are very close in confidence (truly ambiguous).
            if let second = secondBest, margin < 1.05 {
                print("[Classify] VN-only: rejected '\(best.key)' — ambiguous with '\(second.key)' (margin=\(String(format: "%.2f", margin)))")
                return nil
            }

            print("[Classify] VN-only: accepted '\(best.key)' (conf=\(String(format: "%.4f", best.value)), margin=\(String(format: "%.2f", margin)))")
            return ClassificationResult(word: best.key)
        }

        // Full dual-model consensus
        let secondConfidence = secondBest.map { Double($0.value) } ?? 0

        guard let word = consensusGate.evaluate(
            vnClassifyWord: best.key,
            vnConfidence: Double(best.value),
            vnSecondConfidence: secondConfidence,
            clipWord: clip.top1.word,
            clipSimilarity: clip.top1.similarity,
            clipSecondSimilarity: clip.top2.similarity
        ) else {
            return nil
        }

        return ClassificationResult(word: word)
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

    private func runCLIP(_ buffer: CVPixelBuffer) async -> (top1: CLIPResult, top2: CLIPResult)? {
        clipEmbeddings?.classify(imageBuffer: buffer)
    }
}
