import XCTest
@testable import ARYA

final class ConsensusGateTests: XCTestCase {

    let gate = ConsensusGate()

    // MARK: - High confidence custom classifier (>0.90)

    func testCustomHighConfidence_AcceptsWithoutVN() {
        let result = gate.evaluate(
            vnClassifyWord: "window", vnConfidence: 0.05, vnSecondConfidence: 0.02,
            customWord: "monitor", customConfidence: 0.95, customSecondConfidence: 0.03
        )
        XCTAssertEqual(result, "monitor")
    }

    func testCustomHighConfidence_AcceptsEvenWhenVNDisagrees() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.50, vnSecondConfidence: 0.10,
            customWord: "cat", customConfidence: 0.92, customSecondConfidence: 0.04
        )
        XCTAssertEqual(result, "cat")
    }

    // MARK: - Medium confidence (0.4-0.9): needs VN agreement or VN trust

    func testMediumConfidence_BothAgree_Accepts() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.10, vnSecondConfidence: 0.02,
            customWord: "dog", customConfidence: 0.60, customSecondConfidence: 0.15
        )
        XCTAssertEqual(result, "dog")
    }

    func testMediumConfidence_DisagreeWeakVN_Rejects() {
        // VN between 0.20 and 0.45, custom margin < 3 — neither overrides
        let result = gate.evaluate(
            vnClassifyWord: "window", vnConfidence: 0.25, vnSecondConfidence: 0.05,
            customWord: "monitor", customConfidence: 0.55, customSecondConfidence: 0.20
        )
        XCTAssertNil(result)
    }

    func testMediumConfidence_DisagreeStrongVN_TrustsVN() {
        // VN is confident (>0.45) — trust VN over uncertain custom
        let result = gate.evaluate(
            vnClassifyWord: "speaker", vnConfidence: 0.70, vnSecondConfidence: 0.05,
            customWord: "pan", customConfidence: 0.84, customSecondConfidence: 0.04
        )
        XCTAssertEqual(result, "speaker")
    }

    func testMediumConfidence_StrongMarginWeakVN_OverridesVN() {
        // Custom has 4x margin and VN confidence < 0.20
        let result = gate.evaluate(
            vnClassifyWord: "cat", vnConfidence: 0.15, vnSecondConfidence: 0.02,
            customWord: "ear", customConfidence: 0.83, customSecondConfidence: 0.04
        )
        XCTAssertEqual(result, "ear")
    }

    func testMediumConfidence_VNModerateNoOverride_Rejects() {
        // VN is between 0.20-0.45 — neither trusts VN nor allows custom override
        let result = gate.evaluate(
            vnClassifyWord: "moon", vnConfidence: 0.28, vnSecondConfidence: 0.05,
            customWord: "lamp", customConfidence: 0.71, customSecondConfidence: 0.14
        )
        XCTAssertNil(result)
    }

    // MARK: - Low confidence (<0.10): VN fallback

    func testLowConfidence_FallsBackToVN() {
        let result = gate.evaluate(
            vnClassifyWord: "cup", vnConfidence: 0.20, vnSecondConfidence: 0.05,
            customWord: "glass", customConfidence: 0.05, customSecondConfidence: 0.04
        )
        XCTAssertEqual(result, "cup")
    }

    func testBothLowConfidence_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "cup", vnConfidence: 0.02, vnSecondConfidence: 0.01,
            customWord: "glass", customConfidence: 0.05, customSecondConfidence: 0.04
        )
        XCTAssertNil(result)
    }

    // MARK: - Between low and medium (0.10-0.40)

    func testWeakAgreement_Accepts() {
        let result = gate.evaluate(
            vnClassifyWord: "cup", vnConfidence: 0.10, vnSecondConfidence: 0.02,
            customWord: "cup", customConfidence: 0.25, customSecondConfidence: 0.10
        )
        XCTAssertEqual(result, "cup")
    }

    func testWeakCustom_StrongVN_TrustsVN() {
        // Custom is weak (0.10-0.40) but VN is confident (>0.45) — trust VN
        let result = gate.evaluate(
            vnClassifyWord: "cup", vnConfidence: 0.50, vnSecondConfidence: 0.05,
            customWord: "glass", customConfidence: 0.25, customSecondConfidence: 0.10
        )
        XCTAssertEqual(result, "cup")
    }

    func testWeakDisagreement_WeakVN_Rejects() {
        let result = gate.evaluate(
            vnClassifyWord: "cup", vnConfidence: 0.10, vnSecondConfidence: 0.02,
            customWord: "glass", customConfidence: 0.25, customSecondConfidence: 0.10
        )
        XCTAssertNil(result)
    }
}
