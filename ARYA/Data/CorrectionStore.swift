import Foundation
import Accelerate

struct Correction: Codable {
    let word: String
    let embedding: [Float]
}

final class CorrectionStore {
    private var corrections: [Correction] = []
    private let fileURL: URL
    private let similarityThreshold: Float = 0.85

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        fileURL = docs.appendingPathComponent("corrections.json")
        load()
    }

    /// Check if a CLIP embedding matches any stored correction.
    func lookup(embedding: [Float]) -> String? {
        var bestWord: String?
        var bestSimilarity: Float = 0

        for correction in corrections {
            let sim = cosineSimilarity(embedding, correction.embedding)
            if sim > bestSimilarity {
                bestSimilarity = sim
                bestWord = correction.word
            }
        }

        guard bestSimilarity >= similarityThreshold, let word = bestWord else {
            return nil
        }

        print("[Correction] Matched stored correction: '\(word)' (similarity=\(String(format: "%.3f", bestSimilarity)))")
        return word
    }

    /// Save a correction: the CLIP embedding of the image + the correct word.
    func addCorrection(embedding: [Float], word: String) {
        // Remove any existing correction with very similar embedding (update, don't duplicate)
        corrections.removeAll { cosineSimilarity($0.embedding, embedding) >= 0.95 }
        corrections.append(Correction(word: word, embedding: embedding))
        save()
        print("[Correction] Saved correction: '\(word)' (total stored: \(corrections.count))")
    }

    var count: Int { corrections.count }

    private func save() {
        do {
            let data = try JSONEncoder().encode(corrections)
            try data.write(to: fileURL)
        } catch {
            print("[Correction] Failed to save: \(error)")
        }
    }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode([Correction].self, from: data) else {
            return
        }
        corrections = decoded
        print("[Correction] Loaded \(corrections.count) stored corrections")
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        vDSP_dotpr(a, 1, a, 1, &normA, vDSP_Length(a.count))
        vDSP_dotpr(b, 1, b, 1, &normB, vDSP_Length(b.count))
        guard normA > 0 && normB > 0 else { return 0 }
        return dot / (sqrt(normA) * sqrt(normB))
    }
}
