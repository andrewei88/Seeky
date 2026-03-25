import Foundation
import Accelerate

struct Correction: Codable {
    let word: String
    let embedding: [Float]
}

/// Groups all embeddings for a single word, plus a centroid that averages them.
/// The centroid captures the "essence" of the object across viewing angles,
/// so corrections from one angle can extrapolate to others.
private struct WordGroup {
    var embeddings: [[Float]]
    var centroid: [Float]

    init(embedding: [Float]) {
        embeddings = [embedding]
        centroid = embedding
    }

    mutating func addEmbedding(_ embedding: [Float]) {
        embeddings.append(embedding)
        recomputeCentroid()
    }

    /// Remove the most recent embedding. Returns true if the group is now empty.
    mutating func removeLast() -> Bool {
        embeddings.removeLast()
        if embeddings.isEmpty { return true }
        recomputeCentroid()
        return false
    }

    private mutating func recomputeCentroid() {
        let dim = embeddings[0].count
        var sum = [Float](repeating: 0, count: dim)
        for emb in embeddings {
            vDSP_vadd(sum, 1, emb, 1, &sum, 1, vDSP_Length(dim))
        }
        // L2-normalize the centroid so cosine similarity works correctly
        var norm: Float = 0
        vDSP_dotpr(sum, 1, sum, 1, &norm, vDSP_Length(dim))
        norm = sqrt(norm)
        if norm > 0 {
            vDSP_vsdiv(sum, 1, &norm, &sum, 1, vDSP_Length(dim))
        }
        centroid = sum
    }
}

@MainActor
final class CorrectionStore {
    private var corrections: [Correction] = []
    private var wordGroups: [String: WordGroup] = [:]
    private let fileURL: URL
    private let similarityThreshold: Float = 0.80

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        fileURL = docs.appendingPathComponent("corrections.json")
        load()
    }

    /// Check if a feature embedding matches any stored correction.
    /// Checks both individual embeddings and per-word centroids for best coverage.
    func lookup(embedding: [Float]) -> String? {
        var bestWord: String?
        var bestSimilarity: Float = 0
        var matchSource = "individual"

        // Check individual embeddings
        for correction in corrections {
            let sim = cosineSimilarity(embedding, correction.embedding)
            if sim > bestSimilarity {
                bestSimilarity = sim
                bestWord = correction.word
                matchSource = "individual"
            }
        }

        // Check centroids (averaged embeddings capture cross-angle generalization)
        for (word, group) in wordGroups where group.embeddings.count >= 2 {
            let sim = cosineSimilarity(embedding, group.centroid)
            if sim > bestSimilarity {
                bestSimilarity = sim
                bestWord = word
                matchSource = "centroid(\(group.embeddings.count) samples)"
            }
        }

        guard bestSimilarity >= similarityThreshold, let word = bestWord else {
            if bestSimilarity > 0.5 {
                print("[Correction] Near miss: '\(bestWord ?? "?")' (similarity=\(String(format: "%.3f", bestSimilarity)), threshold=\(similarityThreshold))")
            }
            return nil
        }

        print("[Correction] Matched '\(word)' via \(matchSource) (similarity=\(String(format: "%.3f", bestSimilarity)))")
        return word
    }

    /// Save a correction: the feature embedding of the image + the correct word.
    func addCorrection(embedding: [Float], word: String) {
        // Remove any existing correction with very similar embedding (update, don't duplicate)
        let removed = corrections.filter { cosineSimilarity($0.embedding, embedding) >= 0.95 }
        corrections.removeAll { cosineSimilarity($0.embedding, embedding) >= 0.95 }

        // Remove from word groups too
        for r in removed {
            wordGroups[r.word]?.embeddings.removeAll { cosineSimilarity($0, r.embedding) >= 0.95 }
            if wordGroups[r.word]?.embeddings.isEmpty == true {
                wordGroups.removeValue(forKey: r.word)
            }
        }

        corrections.append(Correction(word: word, embedding: embedding))

        // Update word group with centroid
        if wordGroups[word] != nil {
            wordGroups[word]!.addEmbedding(embedding)
            print("[Correction] Saved correction: '\(word)' (\(wordGroups[word]!.embeddings.count) samples, centroid updated)")
        } else {
            wordGroups[word] = WordGroup(embedding: embedding)
            print("[Correction] Saved correction: '\(word)' (first sample)")
        }

        save()
    }

    /// Remove the most recently added correction.
    func undoLastCorrection() -> String? {
        guard let last = corrections.popLast() else { return nil }

        // Update word group
        if let isEmpty = wordGroups[last.word]?.removeLast(), isEmpty {
            wordGroups.removeValue(forKey: last.word)
        }

        save()
        print("[Correction] Undid last correction: '\(last.word)' (total stored: \(corrections.count))")
        return last.word
    }

    /// Remove all stored corrections.
    func clearAll() {
        corrections.removeAll()
        wordGroups.removeAll()
        save()
        print("[Correction] Cleared all corrections")
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

        // Rebuild word groups from stored corrections
        wordGroups.removeAll()
        for correction in corrections {
            if wordGroups[correction.word] != nil {
                wordGroups[correction.word]!.addEmbedding(correction.embedding)
            } else {
                wordGroups[correction.word] = WordGroup(embedding: correction.embedding)
            }
        }

        let groupSummary = wordGroups.map { "\($0.key)(\($0.value.embeddings.count))" }.joined(separator: ", ")
        print("[Correction] Loaded \(corrections.count) corrections across \(wordGroups.count) words: \(groupSummary)")
    }
}
