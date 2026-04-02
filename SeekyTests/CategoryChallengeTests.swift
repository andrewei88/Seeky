import XCTest
@testable import Seeky

@MainActor
final class CategoryChallengeTests: XCTestCase {

    // MARK: - ChallengeTarget & Challenge

    func testWordChallengeMatchesExactWord() {
        let challenge = Challenge(target: .word("cup"))
        XCTAssertTrue(challenge.matches(word: "cup"))
        XCTAssertFalse(challenge.matches(word: "mug"))
        XCTAssertFalse(challenge.matches(word: "Cup"))
    }

    func testCategoryChallengeMatchesMemberWord() {
        let challenge = Challenge(target: .category("animal"))
        XCTAssertTrue(challenge.matches(word: "dog"))
        XCTAssertTrue(challenge.matches(word: "cat"))
        XCTAssertTrue(challenge.matches(word: "elephant"))
        XCTAssertFalse(challenge.matches(word: "cup"))
        XCTAssertFalse(challenge.matches(word: "apple"))
    }

    func testCategoryChallengeUnknownCategoryMatchesNothing() {
        let challenge = Challenge(target: .category("nonexistent"))
        XCTAssertFalse(challenge.matches(word: "dog"))
        XCTAssertFalse(challenge.matches(word: ""))
    }

    func testDisplayTextForWord() {
        let challenge = Challenge(target: .word("banana"))
        XCTAssertEqual(challenge.displayText, "banana")
    }

    func testDisplayTextForCategory() {
        let challenge = Challenge(target: .category("fruit"))
        XCTAssertEqual(challenge.displayText, "fruit")
    }

    // MARK: - All categories have valid members

    func testAllCategoriesHaveAtLeastThreeWords() {
        for (category, words) in WordProgressStore.quizCategories {
            XCTAssertGreaterThanOrEqual(words.count, 3,
                "Category '\(category)' has only \(words.count) words — need at least 3")
        }
    }

    func testAllCategoryWordsAreClassifiable() {
        // Category words must be in the classifier's vocabulary (seeky_classes.json),
        // not necessarily in the seeded quiz pool. The classifier can output any of
        // its 151 classes, and category challenges accept all matches.
        guard let url = Bundle.main.url(forResource: "seeky_classes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let classes = try? JSONDecoder().decode([String].self, from: data) else {
            // Can't load classes in simulator — skip this check
            return
        }
        let classifierVocab = Set(classes)
        for (category, words) in WordProgressStore.quizCategories {
            for word in words {
                XCTAssertTrue(classifierVocab.contains(word),
                    "Word '\(word)' in category '\(category)' is not in classifier vocabulary")
            }
        }
    }

    func testNoCategoryContainsUnsafeWords() {
        for (category, words) in WordProgressStore.quizCategories {
            let unsafe = words.intersection(WordProgressStore.unsafeForQuiz)
            XCTAssertTrue(unsafe.isEmpty,
                "Category '\(category)' contains unsafe words: \(unsafe)")
        }
    }

    // MARK: - QuizSession with challenges

    func testSessionWithMixedChallenges() {
        let challenges = [
            Challenge(target: .word("cup")),
            Challenge(target: .category("animal")),
            Challenge(target: .word("ball")),
        ]
        let session = QuizSession(challenges: challenges)

        XCTAssertEqual(session.challenges.count, 3)
        XCTAssertEqual(session.words, ["cup", "animal", "ball"])
        XCTAssertEqual(session.currentChallenge, challenges[0])
        XCTAssertFalse(session.isComplete)
    }

    func testSessionBackwardsCompatConvenienceInit() {
        let session = QuizSession(words: ["cat", "dog", "fish"])
        XCTAssertEqual(session.challenges.count, 3)

        // All should be word challenges
        for challenge in session.challenges {
            if case .category = challenge.target {
                XCTFail("Convenience init should create word challenges only")
            }
        }
        XCTAssertEqual(session.words, ["cat", "dog", "fish"])
    }

    func testRecordResultTracksfoundWord() {
        let session = QuizSession(challenges: [
            Challenge(target: .category("animal")),
        ])
        session.lastResult = .correct
        session.lastFoundWord = "dog"
        session.recordResult(correct: true, foundWord: "dog")

        XCTAssertEqual(session.results.count, 1)
        XCTAssertEqual(session.results[0].foundWord, "dog")
        XCTAssertTrue(session.results[0].correct)
    }

    func testWrongAnswerStoresActualWord() {
        let session = QuizSession(challenges: [
            Challenge(target: .word("cat")),
        ])
        session.lastResult = .wrong(actual: "dog")
        session.attemptsOnCurrent = 1

        // The actual word is accessible from the result
        if case .wrong(let actual) = session.lastResult {
            XCTAssertEqual(actual, "dog")
        } else {
            XCTFail("Expected wrong result")
        }
    }

    func testWrongAnswerNilActualWhenRejected() {
        let session = QuizSession(challenges: [
            Challenge(target: .word("cat")),
        ])
        session.lastResult = .wrong(actual: nil)

        if case .wrong(let actual) = session.lastResult {
            XCTAssertNil(actual)
        } else {
            XCTFail("Expected wrong result")
        }
    }

    func testHintAvailableAfterTwoAttempts() {
        let session = QuizSession(challenges: [
            Challenge(target: .word("elephant")),
        ])
        // After 2 attempts, hint should show first letter
        session.attemptsOnCurrent = 2
        let targetWord = session.currentChallenge!.displayText
        let hint = String(targetWord.prefix(1)).uppercased()
        XCTAssertEqual(hint, "E")
    }

    func testAdvanceClearsFoundWord() {
        let session = QuizSession(challenges: [
            Challenge(target: .category("animal")),
            Challenge(target: .word("cup")),
        ])
        session.lastFoundWord = "dog"
        session.advance()

        XCTAssertNil(session.lastFoundWord)
        XCTAssertEqual(session.currentIndex, 1)
    }

    // MARK: - Category membership for each category

    func testFruitCategoryMembers() {
        let challenge = Challenge(target: .category("fruit"))
        XCTAssertTrue(challenge.matches(word: "apple"))
        XCTAssertTrue(challenge.matches(word: "banana"))
        XCTAssertTrue(challenge.matches(word: "strawberry"))
        XCTAssertFalse(challenge.matches(word: "pizza"))
    }

    func testFoodCategoryMembers() {
        let challenge = Challenge(target: .category("food"))
        XCTAssertTrue(challenge.matches(word: "bread"))
        XCTAssertTrue(challenge.matches(word: "pizza"))
        XCTAssertFalse(challenge.matches(word: "apple"))
    }

    func testClothingCategoryMembers() {
        let challenge = Challenge(target: .category("clothing"))
        XCTAssertTrue(challenge.matches(word: "hat"))
        XCTAssertTrue(challenge.matches(word: "shoe"))
        XCTAssertFalse(challenge.matches(word: "cup"))
    }

    func testKitchenItemCategoryMembers() {
        let challenge = Challenge(target: .category("kitchen item"))
        XCTAssertTrue(challenge.matches(word: "fork"))
        XCTAssertTrue(challenge.matches(word: "plate"))
        XCTAssertFalse(challenge.matches(word: "knife"), "knife is unsafe for quiz")
        XCTAssertFalse(challenge.matches(word: "bed"))
    }

    func testVehicleCategoryMembers() {
        let challenge = Challenge(target: .category("vehicle"))
        XCTAssertTrue(challenge.matches(word: "car"))
        XCTAssertTrue(challenge.matches(word: "bus"))
        XCTAssertFalse(challenge.matches(word: "horse"))
    }
}
