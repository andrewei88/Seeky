import Foundation
import CoreML
import CoreImage
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
            print("[CLIP] text_embeddings.bin not found in bundle")
            return nil
        }

        let embeddingDim = 512 // MobileCLIP S0 dimension
        let floatCount = data.count / MemoryLayout<Float>.size
        let vectorCount = floatCount / embeddingDim

        guard vectorCount == vocabulary.count else {
            print("[CLIP] Embedding count mismatch: file has \(vectorCount) but vocabulary has \(vocabulary.count)")
            return nil
        }

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
        // Xcode compiles .mlpackage → .mlmodelc at build time
        var model: MLModel?
        if let modelURL = bundle.url(forResource: "MobileCLIPImageEncoder", withExtension: "mlmodelc") {
            do {
                model = try MLModel(contentsOf: modelURL)
                print("[CLIP] Image encoder loaded successfully")
            } catch {
                print("[CLIP] Failed to load image encoder: \(error)")
            }
        } else {
            print("[CLIP] MobileCLIPImageEncoder.mlmodelc not found in bundle")
        }

        return CLIPEmbeddings(vocabulary: vocabulary, embeddings: embeddings, imageEncoder: model)
    }

    /// Encode an image to a CLIP embedding vector (for storing corrections).
    func encode(imageBuffer: CVPixelBuffer) -> [Float]? {
        guard let encoder = imageEncoder else { return nil }
        return encodeImage(imageBuffer, with: encoder)
    }

    /// Classify a cropped image against the vocabulary. Returns top-2 results + embedding.
    func classify(imageBuffer: CVPixelBuffer) -> (top1: CLIPResult, top2: CLIPResult, embedding: [Float])? {
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

        // Log top 5 for debugging
        let top5 = similarities.prefix(5).map { "\($0.word)(\(String(format: "%.3f", $0.similarity)))" }
        print("[CLIP] Top 5: \(top5.joined(separator: ", "))")

        return (
            top1: CLIPResult(word: similarities[0].word, similarity: similarities[0].similarity),
            top2: CLIPResult(word: similarities[1].word, similarity: similarities[1].similarity),
            embedding: imageEmbedding
        )
    }

    private func encodeImage(_ buffer: CVPixelBuffer, with model: MLModel) -> [Float]? {
        // Resize to 256x256 as required by MobileCLIP S0
        guard let resizedBuffer = resizePixelBuffer(buffer, to: CGSize(width: 256, height: 256)) else {
            print("[CLIP] Failed to resize buffer to 256x256")
            return nil
        }

        // Create MLFeatureValue from pixel buffer
        guard let input = try? MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: resizedBuffer)]),
              let output = try? model.prediction(from: input),
              let embeddingFeature = output.featureValue(for: "final_emb_1"),
              let multiArray = embeddingFeature.multiArrayValue else {
            print("[CLIP] Model prediction failed")
            return nil
        }

        let count = multiArray.count
        var result = [Float](repeating: 0, count: count)
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            result[i] = ptr[i]
        }

        // L2 normalize the embedding
        var norm: Float = 0
        vDSP_dotpr(result, 1, result, 1, &norm, vDSP_Length(count))
        norm = sqrt(norm)
        if norm > 0 {
            var scale = 1.0 / norm
            vDSP_vsmul(result, 1, &scale, &result, 1, vDSP_Length(count))
        }

        return result
    }

    /// Resize a CVPixelBuffer to the target size using CIImage.
    private func resizePixelBuffer(_ buffer: CVPixelBuffer, to size: CGSize) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let scaleX = size.width / ciImage.extent.width
        let scaleY = size.height / ciImage.extent.height
        let resized = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        let context = CIContext()
        var outputBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height),
                           kCVPixelFormatType_32BGRA, nil, &outputBuffer)
        guard let output = outputBuffer else { return nil }
        context.render(resized, to: output)
        return output
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
