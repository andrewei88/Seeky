import XCTest
@testable import ARYA

final class ConsensusGateTests: XCTestCase {

    let gate = ConsensusGate()

    func testBothAgreeHighConfidence_Accepts() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.90, vnSecondConfidence: 0.10,
            clipWord: "dog", clipSimilarity: 0.85, clipSecondSimilarity: 0.30
        )
        XCTAssertEqual(result, "dog")
    }

    func testDisagreement_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.90, vnSecondConfidence: 0.10,
            clipWord: "cat", clipSimilarity: 0.85, clipSecondSimilarity: 0.30
        )
        XCTAssertNil(result)
    }

    func testVNConfidenceTooLow_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.50, vnSecondConfidence: 0.10,
            clipWord: "dog", clipSimilarity: 0.85, clipSecondSimilarity: 0.30
        )
        XCTAssertNil(result)
    }

    func testCLIPSimilarityTooLow_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.90, vnSecondConfidence: 0.10,
            clipWord: "dog", clipSimilarity: 0.60, clipSecondSimilarity: 0.30
        )
        XCTAssertNil(result)
    }

    func testVNMarginTooSmall_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.80, vnSecondConfidence: 0.60,
            clipWord: "dog", clipSimilarity: 0.85, clipSecondSimilarity: 0.30
        )
        XCTAssertNil(result)
    }

    func testCLIPMarginTooSmall_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.90, vnSecondConfidence: 0.10,
            clipWord: "dog", clipSimilarity: 0.80, clipSecondSimilarity: 0.70
        )
        XCTAssertNil(result)
    }

    func testNilVNWord_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: nil, vnConfidence: 0.90, vnSecondConfidence: 0.10,
            clipWord: "dog", clipSimilarity: 0.85, clipSecondSimilarity: 0.30
        )
        XCTAssertNil(result)
    }
}
