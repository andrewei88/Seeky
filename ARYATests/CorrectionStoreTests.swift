import XCTest
@testable import ARYA

final class CorrectionStoreTests: XCTestCase {

    func testAddAndLookupCorrection() {
        let store = CorrectionStore()

        // Create a fake embedding (unit vector)
        var embedding = [Float](repeating: 0, count: 512)
        embedding[0] = 1.0

        store.addCorrection(embedding: embedding, word: "cat")

        // Same embedding should match
        let result = store.lookup(embedding: embedding)
        XCTAssertEqual(result, "cat")
    }

    func testSimilarEmbeddingMatches() {
        let store = CorrectionStore()

        var embedding = [Float](repeating: 0, count: 512)
        embedding[0] = 1.0
        store.addCorrection(embedding: embedding, word: "dog")

        // Slightly different embedding — high cosine similarity
        var similar = [Float](repeating: 0, count: 512)
        similar[0] = 0.99
        similar[1] = 0.1
        let result = store.lookup(embedding: similar)
        XCTAssertEqual(result, "dog")
    }

    func testDissimilarEmbeddingDoesNotMatch() {
        let store = CorrectionStore()

        var embedding = [Float](repeating: 0, count: 512)
        embedding[0] = 1.0
        store.addCorrection(embedding: embedding, word: "cat")

        // Orthogonal embedding — cosine similarity = 0
        var orthogonal = [Float](repeating: 0, count: 512)
        orthogonal[1] = 1.0
        let result = store.lookup(embedding: orthogonal)
        XCTAssertNil(result)
    }

    func testDuplicateEmbeddingUpdatesWord() {
        let store = CorrectionStore()

        var embedding = [Float](repeating: 0, count: 512)
        embedding[0] = 1.0

        store.addCorrection(embedding: embedding, word: "cat")
        store.addCorrection(embedding: embedding, word: "dog")

        // Should return latest correction, not first
        let result = store.lookup(embedding: embedding)
        XCTAssertEqual(result, "dog")

        // Should not have duplicate entries
        XCTAssertEqual(store.count, 1)
    }
}
