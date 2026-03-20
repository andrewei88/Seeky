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

        let vn = await vnResult
        let clip = await clipResult

        guard let vn = vn else { return nil }

        // Map VN label to child word
        let vnChildWord = labelMapper.childWord(for: vn.label)

        // If no CLIP available, use VN alone with stricter threshold
        guard let clip = clip else {
            // Fallback: VN only with very high confidence
            guard let word = vnChildWord,
                  vn.confidence >= 0.85,
                  vn.secondConfidence == 0 || vn.confidence >= vn.secondConfidence * 2.0 else {
                return nil
            }
            return ClassificationResult(word: word)
        }

        // Full dual-model consensus
        guard let word = consensusGate.evaluate(
            vnClassifyWord: vnChildWord,
            vnConfidence: Double(vn.confidence),
            vnSecondConfidence: Double(vn.secondConfidence),
            clipWord: clip.top1.word,
            clipSimilarity: clip.top1.similarity,
            clipSecondSimilarity: clip.top2.similarity
        ) else {
            return nil
        }

        return ClassificationResult(word: word)
    }

    private struct VNResult {
        let label: String
        let confidence: Float
        let secondConfidence: Float
    }

    private func runVNClassify(_ buffer: CVPixelBuffer) async -> VNResult? {
        await withCheckedContinuation { continuation in
            let request = VNClassifyImageRequest { request, error in
                guard let observations = request.results as? [VNClassificationObservation],
                      observations.count >= 2 else {
                    continuation.resume(returning: nil)
                    return
                }
                continuation.resume(returning: VNResult(
                    label: observations[0].identifier,
                    confidence: observations[0].confidence,
                    secondConfidence: observations[1].confidence
                ))
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
