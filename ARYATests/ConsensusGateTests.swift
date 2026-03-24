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
        // VN at 0.50 is below the 0.80 override threshold — custom still wins
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.50, vnSecondConfidence: 0.10,
            customWord: "cat", customConfidence: 0.92, customSecondConfidence: 0.04
        )
        XCTAssertEqual(result, "cat")
    }

    func testCustomHighConfidence_VNVeryStrongDisagree_TrustsVN() {
        // The speaker→"key" pattern: custom 0.920 says "key" but VN "speaker" at 1.12
        // VN >= 0.80 overrides even high-confidence custom
        let result = gate.evaluate(
            vnClassifyWord: "speaker", vnConfidence: 1.12, vnSecondConfidence: 0.05,
            customWord: "key", customConfidence: 0.920, customSecondConfidence: 0.037
        )
        XCTAssertEqual(result, "speaker")
    }

    // MARK: - Medium confidence (0.4-0.9): needs VN agreement or VN trust

    func testMediumConfidence_BothAgree_Accepts() {
        let result = gate.evaluate(
            vnClassifyWord: "dog", vnConfidence: 0.10, vnSecondConfidence: 0.02,
            customWord: "dog", customConfidence: 0.60, customSecondConfidence: 0.15
        )
        XCTAssertEqual(result, "dog")
    }

    func testMediumConfidence_DisagreeWeakVN_StrongMargin_AcceptsCustom() {
        // VN untrusted (<0.45), custom margin 2.75 ≥ 2.0 → trust custom
        let result = gate.evaluate(
            vnClassifyWord: "window", vnConfidence: 0.25, vnSecondConfidence: 0.05,
            customWord: "monitor", customConfidence: 0.55, customSecondConfidence: 0.20
        )
        XCTAssertEqual(result, "monitor")
    }

    func testMediumConfidence_DisagreeWeakVN_WeakMargin_Rejects() {
        // VN untrusted (<0.45), but custom margin 1.5 < 2.0 → reject
        let result = gate.evaluate(
            vnClassifyWord: "window", vnConfidence: 0.25, vnSecondConfidence: 0.05,
            customWord: "monitor", customConfidence: 0.45, customSecondConfidence: 0.30
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

    func testVeryStrongCustom_OverridesMarginalVN() {
        // The speaker bug: custom speaker(0.732) margin 7.3 vs VN bottle(0.459)
        // VN barely clears trust threshold but custom is overwhelmingly confident → trust custom
        let result = gate.evaluate(
            vnClassifyWord: "bottle", vnConfidence: 0.459, vnSecondConfidence: 0.05,
            customWord: "speaker", customConfidence: 0.732, customSecondConfidence: 0.10
        )
        XCTAssertEqual(result, "speaker")
    }

    func testVeryStrongCustom_DoesNotOverrideStrongVN() {
        // Custom is strong but VN is clearly confident (>=0.60) → trust VN
        let result = gate.evaluate(
            vnClassifyWord: "bottle", vnConfidence: 0.65, vnSecondConfidence: 0.05,
            customWord: "speaker", customConfidence: 0.75, customSecondConfidence: 0.10
        )
        XCTAssertEqual(result, "bottle")
    }

    func testStrongCustom_WeakMargin_DoesNotOverrideMarginalVN() {
        // Custom is confident but margin is only 2.0 (not >=5.0) → VN still wins at 0.48
        let result = gate.evaluate(
            vnClassifyWord: "bottle", vnConfidence: 0.48, vnSecondConfidence: 0.05,
            customWord: "speaker", customConfidence: 0.72, customSecondConfidence: 0.36
        )
        XCTAssertEqual(result, "bottle")
    }

    func testMediumConfidence_StrongMarginWeakVN_OverridesVN() {
        // Custom has 20x margin and VN untrusted (<0.45)
        let result = gate.evaluate(
            vnClassifyWord: "cat", vnConfidence: 0.15, vnSecondConfidence: 0.02,
            customWord: "ear", customConfidence: 0.83, customSecondConfidence: 0.04
        )
        XCTAssertEqual(result, "ear")
    }

    func testMediumConfidence_VNModerate_StrongCustomMargin_AcceptsCustom() {
        // The lamp scenario: custom "lamp" at 0.71 with 5x margin, VN "moon" at 0.28 untrusted
        // Custom margin 5.07 ≥ 2.0 → trust custom
        let result = gate.evaluate(
            vnClassifyWord: "moon", vnConfidence: 0.28, vnSecondConfidence: 0.05,
            customWord: "lamp", customConfidence: 0.71, customSecondConfidence: 0.14
        )
        XCTAssertEqual(result, "lamp")
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
