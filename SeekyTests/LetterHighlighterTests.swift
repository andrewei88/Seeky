import XCTest
@testable import Seeky

final class LetterHighlighterTests: XCTestCase {

    func makeTimingData() -> TimingData {
        let json = """
        {
          "word": "dog",
          "phonemes": [
            { "phoneme": "D",  "letters": [0], "start": 0.00, "end": 0.40 },
            { "phoneme": "AO", "letters": [1], "start": 0.40, "end": 0.95 },
            { "phoneme": "G",  "letters": [2], "start": 0.95, "end": 1.40 }
          ]
        }
        """.data(using: .utf8)!
        return try! JSONDecoder().decode(TimingData.self, from: json)
    }

    func testAllLettersStartDim() {
        let highlighter = LetterHighlighter(timing: makeTimingData())
        let states = highlighter.letterStates(at: -0.1)
        XCTAssertEqual(states.count, 3)
        XCTAssertTrue(states.allSatisfy { $0 == .upcoming })
    }

    func testFirstLetterActiveAtStart() {
        let highlighter = LetterHighlighter(timing: makeTimingData())
        let states = highlighter.letterStates(at: 0.20)
        XCTAssertEqual(states[0], .active)
        XCTAssertEqual(states[1], .upcoming)
        XCTAssertEqual(states[2], .upcoming)
    }

    func testMiddleLetterActive() {
        let highlighter = LetterHighlighter(timing: makeTimingData())
        let states = highlighter.letterStates(at: 0.60)
        XCTAssertEqual(states[0], .spoken)
        XCTAssertEqual(states[1], .active)
        XCTAssertEqual(states[2], .upcoming)
    }

    func testAllSpokenAfterEnd() {
        let highlighter = LetterHighlighter(timing: makeTimingData())
        let states = highlighter.letterStates(at: 1.50)
        XCTAssertTrue(states.allSatisfy { $0 == .spoken })
    }

    func testMultiLetterPhoneme() {
        let json = """
        {
          "word": "elephant",
          "phonemes": [
            { "phoneme": "EH", "letters": [0], "start": 0.00, "end": 0.28 },
            { "phoneme": "L",  "letters": [1], "start": 0.28, "end": 0.52 },
            { "phoneme": "AH", "letters": [2], "start": 0.52, "end": 0.78 },
            { "phoneme": "F",  "letters": [3, 4], "start": 0.78, "end": 1.10 },
            { "phoneme": "AH", "letters": [5], "start": 1.10, "end": 1.35 },
            { "phoneme": "N",  "letters": [6], "start": 1.35, "end": 1.62 },
            { "phoneme": "T",  "letters": [7], "start": 1.62, "end": 1.85 }
          ]
        }
        """.data(using: .utf8)!
        let timing = try! JSONDecoder().decode(TimingData.self, from: json)
        let highlighter = LetterHighlighter(timing: timing)

        // At 0.90s, "ph" (indices 3 and 4) should both be active
        let states = highlighter.letterStates(at: 0.90)
        XCTAssertEqual(states[3], .active)
        XCTAssertEqual(states[4], .active)
        XCTAssertEqual(states[2], .spoken)
        XCTAssertEqual(states[5], .upcoming)
    }

    func testWordWithSpace() {
        let json = """
        {
          "word": "teddy bear",
          "phonemes": [
            { "phoneme": "T",  "letters": [0], "start": 0.00, "end": 0.20 },
            { "phoneme": "EH", "letters": [1], "start": 0.20, "end": 0.45 },
            { "phoneme": "D",  "letters": [2, 3], "start": 0.45, "end": 0.70 },
            { "phoneme": "IY", "letters": [4], "start": 0.70, "end": 1.00 },
            { "phoneme": "B",  "letters": [6], "start": 1.10, "end": 1.30 },
            { "phoneme": "EH", "letters": [7], "start": 1.30, "end": 1.55 },
            { "phoneme": "R",  "letters": [8, 9], "start": 1.55, "end": 1.85 }
          ]
        }
        """.data(using: .utf8)!
        let timing = try! JSONDecoder().decode(TimingData.self, from: json)
        let highlighter = LetterHighlighter(timing: timing)

        // "teddy bear" has 10 characters (index 5 is space)
        let states = highlighter.letterStates(at: 1.20)
        XCTAssertEqual(states.count, 10)
        XCTAssertEqual(states[5], .space) // space stays neutral
        XCTAssertEqual(states[6], .active)
    }
}
