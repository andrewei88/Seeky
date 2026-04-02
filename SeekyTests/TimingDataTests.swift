import XCTest
@testable import Seeky

final class TimingDataTests: XCTestCase {

    func testDecodesTimingJSON() throws {
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

        let timing = try JSONDecoder().decode(TimingData.self, from: json)
        XCTAssertEqual(timing.word, "dog")
        XCTAssertEqual(timing.phonemes.count, 3)
        XCTAssertEqual(timing.phonemes[0].phoneme, "D")
        XCTAssertEqual(timing.phonemes[0].letters, [0])
        XCTAssertEqual(timing.phonemes[0].start, 0.0, accuracy: 0.001)
        XCTAssertEqual(timing.phonemes[0].end, 0.40, accuracy: 0.001)
    }

    func testMultiLetterPhoneme() throws {
        let json = """
        {
          "word": "elephant",
          "phonemes": [
            { "phoneme": "F", "letters": [3, 4], "start": 0.78, "end": 1.10 }
          ]
        }
        """.data(using: .utf8)!

        let timing = try JSONDecoder().decode(TimingData.self, from: json)
        XCTAssertEqual(timing.phonemes[0].letters, [3, 4])
    }

    func testActiveLettersAtTime() throws {
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

        let timing = try JSONDecoder().decode(TimingData.self, from: json)
        XCTAssertEqual(timing.activeLetterIndices(at: 0.20), [0])
        XCTAssertEqual(timing.activeLetterIndices(at: 0.60), [1])
        XCTAssertEqual(timing.activeLetterIndices(at: 1.10), [2])
        XCTAssertTrue(timing.activeLetterIndices(at: 1.50).isEmpty)
    }

    func testSpokenLetterIndicesAtTime() throws {
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

        let timing = try JSONDecoder().decode(TimingData.self, from: json)
        XCTAssertEqual(timing.spokenLetterIndices(at: 0.60), [0])
        XCTAssertEqual(timing.spokenLetterIndices(at: 1.10), [0, 1])
    }
}
