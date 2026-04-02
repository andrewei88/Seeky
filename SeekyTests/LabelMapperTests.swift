import XCTest
@testable import Seeky

final class LabelMapperTests: XCTestCase {

    func testMapsDetailedLabelToChildWord() {
        let mappings: [String: String?] = [
            "golden retriever": "dog",
            "sedan": "car",
            "coffee mug": "cup"
        ]
        let mapper = LabelMapper(mappings: mappings)
        XCTAssertEqual(mapper.childWord(for: "golden retriever"), "dog")
        XCTAssertEqual(mapper.childWord(for: "sedan"), "car")
        XCTAssertEqual(mapper.childWord(for: "coffee mug"), "cup")
    }

    func testReturnsNilForUnmappedLabel() {
        let mapper = LabelMapper(mappings: [:])
        XCTAssertNil(mapper.childWord(for: "espresso machine"))
        XCTAssertNil(mapper.childWord(for: "unknown thing"))
    }

    func testReturnsNilForExplicitlyRejectedLabel() {
        let mappings: [String: String?] = ["espresso": nil]
        let mapper = LabelMapper(mappings: mappings)
        XCTAssertNil(mapper.childWord(for: "espresso"))
    }
}
