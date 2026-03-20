import Foundation
import CoreML
import Accelerate

struct CLIPResult {
    let word: String
    let similarity: Double
}

final class CLIPEmbeddings {
    private let vocabulary: [String]
    private let embeddings: [[Float]] // One embedding vector per vocabulary word
    private var imageEncoder: MLModel?

    init(vocabulary: [String], embeddings: [[Float]], imageEncoder: MLModel? = nil) {
        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.imageEncoder = imageEncoder
    }

    /// Load pre-computed text embeddings from binary file.
    static func load(vocabulary: [String], from bundle: Bundle = .main) -> CLIPEmbeddings? {
        guard let url = bundle.url(forResource: "text_embeddings", withExtension: "bin"),
              let data = try? Data(contentsOf: url) else {
            return nil
        }

        let embeddingDim = 512 // MobileCLIP S2 dimension
        let floatCount = data.count / MemoryLayout<Float>.size
        let vectorCount = floatCount / embeddingDim

        guard vectorCount == vocabulary.count else { return nil }

        var allFloats = [Float](repeating: 0, count: floatCount)
        data.withUnsafeBytes { ptr in
            allFloats = Array(ptr.bindMemory(to: Float.self))
        }

        var embeddings: [[Float]] = []
        for i in 0..<vectorCount {
            let start = i * embeddingDim
            let vector = Array(allFloats[start..<start + embeddingDim])
            embeddings.append(vector)
        }

        // Try to load the MobileCLIP image encoder model
        let modelURL = bundle.url(forResource: "MobileCLIPImageEncoder", withExtension: "mlmodelc")
        let model = modelURL.flatMap { try? MLModel(contentsOf: $0) }

        return CLIPEmbeddings(vocabulary: vocabulary, embeddings: embeddings, imageEncoder: model)
    }

    /// Classify a cropped image against the vocabulary. Returns top-2 results.
    func classify(imageBuffer: CVPixelBuffer) -> (top1: CLIPResult, top2: CLIPResult)? {
        guard let encoder = imageEncoder else { return nil }

        // Run image through MobileCLIP image encoder
        guard let imageEmbedding = encodeImage(imageBuffer, with: encoder) else { return nil }

        // Compute cosine similarity against all vocabulary embeddings
        var similarities: [(word: String, similarity: Double)] = []
        for (index, textEmbedding) in embeddings.enumerated() {
            let sim = cosineSimilarity(imageEmbedding, textEmbedding)
            similarities.append((vocabulary[index], Double(sim)))
        }

        similarities.sort { $0.similarity > $1.similarity }

        guard similarities.count >= 2 else { return nil }

        return (
            top1: CLIPResult(word: similarities[0].word, similarity: similarities[0].similarity),
            top2: CLIPResult(word: similarities[1].word, similarity: similarities[1].similarity)
        )
    }

    private func encodeImage(_ buffer: CVPixelBuffer, with model: MLModel) -> [Float]? {
        // Create MLFeatureValue from pixel buffer
        guard let input = try? MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)]),
              let output = try? model.prediction(from: input),
              let embeddingFeature = output.featureValue(for: "embedding"),
              let multiArray = embeddingFeature.multiArrayValue else {
            return nil
        }

        let count = multiArray.count
        var result = [Float](repeating: 0, count: count)
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            result[i] = ptr[i]
        }
        return result
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
        vDSP_dotpr(a, 1, a, 1, &normA, vDSP_Length(a.count))
        vDSP_dotpr(b, 1, b, 1, &normB, vDSP_Length(b.count))
        guard normA > 0 && normB > 0 else { return 0 }
        return dotProduct / (sqrt(normA) * sqrt(normB))
    }
}
