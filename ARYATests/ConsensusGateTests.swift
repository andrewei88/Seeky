import XCTest
@testable import ARYA

final class ConsensusGateTests: XCTestCase {

    let gate = ConsensusGate()

    func testBothAgree_Accepts() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.10, vnSecondConfidence: 0.02,
            clipWord: "dog", clipSimilarity: 0.30, clipSecondSimilarity: 0.20
        )
        XCTAssertEqual(result, "dog")
    }

    func testDisagreement_CLIPOverrides() {
        // CLIP has good similarity and margin → trusts CLIP over VN
        let result = gate.evaluate(
            vnClassifyWord: "window", vnConfidence: 0.05, vnSecondConfidence: 0.02,
            clipWord: "monitor", clipSimilarity: 0.30, clipSecondSimilarity: 0.20
        )
        XCTAssertEqual(result, "monitor")
    }

    func testDisagreement_CLIPTooLow_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "window", vnConfidence: 0.05, vnSecondConfidence: 0.02,
            clipWord: "monitor", clipSimilarity: 0.15, clipSecondSimilarity: 0.10
        )
        XCTAssertNil(result)
    }

    func testDisagreement_CLIPMarginTooNarrow_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "window", vnConfidence: 0.05, vnSecondConfidence: 0.02,
            clipWord: "monitor", clipSimilarity: 0.25, clipSecondSimilarity: 0.24
        )
        XCTAssertNil(result)
    }

    func testCLIPSimilarityBelowThreshold_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.10, vnSecondConfidence: 0.02,
            clipWord: "dog", clipSimilarity: 0.10, clipSecondSimilarity: 0.05
        )
        XCTAssertNil(result)
    }

    func testAgreementWithHighConfidence() {
        let result = gate.evaluate(
            vnClassifyWord: "cup", vnConfidence: 1.20, vnSecondConfidence: 0.10,
            clipWord: "cup", clipSimilarity: 0.45, clipSecondSimilarity: 0.20
        )
        XCTAssertEqual(result, "cup")
    }
}
