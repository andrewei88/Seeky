import XCTest
@testable import Seeky

@MainActor
final class WordProgressStoreTests: XCTestCase {

    private func freshStore() -> WordProgressStore {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_progress_\(UUID().uuidString).json")
        return WordProgressStore(fileURL: tmp)
    }

    // MARK: - Explore tracking

    func testExploreIdentification() {
        let store = freshStore()
        store.recordExploreIdentification(word: "cat")

        let p = store.progress["cat"]
        XCTAssertNotNil(p)
        XCTAssertEqual(p?.exploreIdentified, 1)
        XCTAssertEqual(p?.exploreCorrected, 0)
        XCTAssertEqual(store.wordsSeen, 1)
    }

    func testExploreCorrection() {
        let store = freshStore()
        store.recordExploreIdentification(word: "dog")
        store.recordExploreCorrection(word: "dog")

        let p = store.progress["dog"]!
        XCTAssertEqual(p.exploreIdentified, 1)
        XCTAssertEqual(p.exploreCorrected, 1)
    }

    // MARK: - Quiz eligibility

    func testQuizPoolIsSeededWordsMinusUnsafe() {
        let store = freshStore()
        let expected = WordProgressStore.seededHighConfidenceWords
            .subtracting(WordProgressStore.unsafeForQuiz)

        // Quiz pool equals seeded words minus unsafe
        let eligible = Set(store.quizEligibleWords())
        XCTAssertEqual(eligible, expected)
        XCTAssertFalse(eligible.contains("sun"), "sun should be excluded (unsafe)")

        // Explore-proven words do NOT enter the quiz pool (disabled for accuracy)
        store.recordExploreIdentification(word: "dinosaur")
        store.recordExploreIdentification(word: "dinosaur")
        let afterExplore = Set(store.quizEligibleWords())
        XCTAssertFalse(afterExplore.contains("dinosaur"))
        XCTAssertEqual(afterExplore, expected)
    }

    // MARK: - Quiz tracking

    func testQuizCorrectIncrementsStreak() {
        let store = freshStore()
        store.recordQuizCorrect(word: "cat")

        let p = store.progress["cat"]!
        XCTAssertEqual(p.quizCorrect, 1)
        XCTAssertEqual(p.quizConsecutiveCorrect, 1)
        XCTAssertEqual(p.masteryLevel, 1)
        XCTAssertNotNil(p.quizLastDate)
    }

    func testQuizWrongResetsStreak() {
        let store = freshStore()
        store.recordQuizCorrect(word: "dog")
        store.recordQuizCorrect(word: "dog")
        XCTAssertEqual(store.progress["dog"]?.quizConsecutiveCorrect, 2)

        store.recordQuizWrong(word: "dog")
        let p = store.progress["dog"]!
        XCTAssertEqual(p.quizConsecutiveCorrect, 0)
        XCTAssertEqual(p.quizWrong, 1)
        XCTAssertEqual(p.masteryLevel, 1) // dropped from 2 to 1
    }

    // MARK: - Mastery progression

    func testMasteryLevelProgression() {
        XCTAssertEqual(WordProgressStore.masteryLevel(for: 0), 0)
        XCTAssertEqual(WordProgressStore.masteryLevel(for: 1), 1)
        XCTAssertEqual(WordProgressStore.masteryLevel(for: 2), 2)
        XCTAssertEqual(WordProgressStore.masteryLevel(for: 3), 2)
        XCTAssertEqual(WordProgressStore.masteryLevel(for: 4), 3)
        XCTAssertEqual(WordProgressStore.masteryLevel(for: 7), 3)
        XCTAssertEqual(WordProgressStore.masteryLevel(for: 8), 4)
        XCTAssertEqual(WordProgressStore.masteryLevel(for: 20), 4)
    }

    func testMasteryReachesLevel4() {
        let store = freshStore()
        for _ in 0..<8 {
            store.recordQuizCorrect(word: "ball")
        }
        XCTAssertEqual(store.progress["ball"]?.masteryLevel, 4)
        XCTAssertEqual(store.wordsMastered, 1)
    }

    func testQuizWrongDropsMastery() {
        let store = freshStore()
        for _ in 0..<8 {
            store.recordQuizCorrect(word: "cup")
        }
        XCTAssertEqual(store.progress["cup"]?.masteryLevel, 4)

        store.recordQuizWrong(word: "cup")
        XCTAssertEqual(store.progress["cup"]?.masteryLevel, 3)
        XCTAssertEqual(store.progress["cup"]?.quizConsecutiveCorrect, 0)
    }

    // MARK: - Quiz word selection

    func testSelectQuizWordsReturnsFromSeededPool() {
        let store = freshStore()
        // Even with no explore history, seeded words provide a quiz pool
        let selected = store.selectQuizWords()
        XCTAssertEqual(selected.count, 5)
        let seeded = WordProgressStore.seededHighConfidenceWords
        for word in selected {
            XCTAssertTrue(seeded.contains(word), "'\(word)' should be a seeded word")
        }
    }

    func testSelectQuizWordsCapsAtCount() {
        let store = freshStore()
        let selected = store.selectQuizWords(count: 3)
        XCTAssertEqual(selected.count, 3)
    }

    // MARK: - Stats

    func testWordsNeedingPractice() {
        let store = freshStore()
        store.recordQuizWrong(word: "cat")  // mastery 0 → needs practice
        store.recordQuizCorrect(word: "dog")
        store.recordQuizCorrect(word: "dog") // mastery 2 → not needing practice

        XCTAssertEqual(store.wordsNeedingPractice, 1) // only cat
    }

    func testMasteryBreakdown() {
        let store = freshStore()
        store.recordQuizCorrect(word: "cat")  // mastery 1
        store.recordQuizCorrect(word: "dog")
        store.recordQuizCorrect(word: "dog")  // mastery 2

        let breakdown = store.masteryBreakdown()
        XCTAssertEqual(breakdown.count, 2)
        XCTAssertEqual(breakdown[0].level, 1)
        XCTAssertTrue(breakdown[0].words.contains("cat"))
        XCTAssertEqual(breakdown[1].level, 2)
        XCTAssertTrue(breakdown[1].words.contains("dog"))
    }

    func testMasteryBreakdownExcludesUnquizzedWords() {
        let store = freshStore()
        store.recordExploreIdentification(word: "cat") // only explored, never quizzed
        store.recordQuizCorrect(word: "dog") // quizzed

        let breakdown = store.masteryBreakdown()
        XCTAssertEqual(breakdown.count, 1)
        XCTAssertTrue(breakdown[0].words.contains("dog"))
    }

    // MARK: - Clear

    func testClearAll() {
        let store = freshStore()
        store.recordExploreIdentification(word: "cat")
        store.recordQuizCorrect(word: "dog")
        XCTAssertTrue(store.wordsSeen > 0)

        store.clearAll()
        XCTAssertEqual(store.wordsSeen, 0)
        XCTAssertEqual(store.wordsMastered, 0)
        // Seeded words (minus unsafe) remain in the quiz pool even after clearing
        let expectedPool = WordProgressStore.seededHighConfidenceWords.subtracting(WordProgressStore.unsafeForQuiz)
        XCTAssertEqual(store.quizPoolSize, expectedPool.count)
    }

    // MARK: - Location-aware selection

    func testSelectQuizWordsLocationFiltering() {
        let store = freshStore()
        let selected = store.selectQuizWords(count: 5, location: .home)
        XCTAssertEqual(selected.count, 5)
        let homeWords = WordProgressStore.locationWords[.home] ?? []
        for word in selected {
            XCTAssertTrue(homeWords.contains(word),
                          "'\(word)' not in home location words")
        }
    }

    func testSelectQuizWordsNilLocationUsesFullPool() {
        let store = freshStore()
        let selected = store.selectQuizWords(count: 5)
        XCTAssertEqual(selected.count, 5)
    }

    func testSkippedWordsDePrioritizedAcrossSessions() {
        let store = freshStore()
        // Simulate: select 5 words from furniture category, skip all of them
        let session1 = store.selectQuizWords(count: 5, categories: Set(["furniture"]))
        XCTAssertEqual(session1.count, 5)
        for word in session1 {
            store.recordQuizSkip(word: word)
        }

        // Session 2: should prefer unseen furniture words over just-skipped ones
        let session2 = store.selectQuizWords(count: 5, categories: Set(["furniture"]))
        XCTAssertEqual(session2.count, 5)

        // At least some words should be different (drawn from the unseen portion)
        let overlap = Set(session1).intersection(Set(session2))
        let furniturePool = WordProgressStore.quizCategories["furniture"]!
            .intersection(WordProgressStore.seededHighConfidenceWords)
            .subtracting(WordProgressStore.unsafeForQuiz)
        // If pool > session size, we should get at least some new words
        if furniturePool.count > 5 {
            XCTAssertTrue(overlap.count < 5,
                "Session 2 should include at least one word not in session 1 (overlap: \(overlap))")
        }
    }

    func testAllSeededWordsHaveLocation() {
        let seeded = WordProgressStore.seededHighConfidenceWords
            .subtracting(WordProgressStore.unsafeForQuiz)
        let allLocationWords = Set(WordProgressStore.locationWords.values.flatMap { $0 })
        let untagged = seeded.subtracting(allLocationWords)
        XCTAssertTrue(untagged.isEmpty,
                      "Seeded words missing from all locations: \(untagged.sorted())")
    }

    // MARK: - Persistence

    func testPersistence() {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_progress_persist_\(UUID().uuidString).json")

        let store1 = WordProgressStore(fileURL: tmp)
        store1.recordExploreIdentification(word: "ball")
        store1.recordExploreIdentification(word: "ball")
        store1.recordQuizCorrect(word: "ball")

        let store2 = WordProgressStore(fileURL: tmp)
        XCTAssertEqual(store2.progress["ball"]?.exploreIdentified, 2)
        XCTAssertEqual(store2.progress["ball"]?.quizCorrect, 1)
        XCTAssertEqual(store2.progress["ball"]?.masteryLevel, 1)

        try? FileManager.default.removeItem(at: tmp)
    }
}
