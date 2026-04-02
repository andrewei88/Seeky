import XCTest
@testable import Seeky

final class TrainingCaptureTests: XCTestCase {

    private func freshCapture() -> TrainingCapture {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_captures_\(UUID().uuidString)")
        return TrainingCapture(baseDir: tmp)
    }

    private func createTestImage(in dir: URL, word: String, count: Int) {
        let wordDir = dir.appendingPathComponent(word.replacingOccurrences(of: " ", with: "_"))
        try! FileManager.default.createDirectory(at: wordDir, withIntermediateDirectories: true)
        for i in 0..<count {
            let fileURL = wordDir.appendingPathComponent("\(i).jpg")
            // Write a minimal valid JPEG (just enough bytes to be a file)
            let data = Data([0xFF, 0xD8, 0xFF, 0xE0])
            try! data.write(to: fileURL)
        }
    }

    override func tearDown() {
        super.tearDown()
        // Clean up any temp directories
        let tmp = FileManager.default.temporaryDirectory
        let contents = (try? FileManager.default.contentsOfDirectory(at: tmp, includingPropertiesForKeys: nil)) ?? []
        for url in contents where url.lastPathComponent.hasPrefix("test_captures_") {
            try? FileManager.default.removeItem(at: url)
        }
    }

    // MARK: - Stats

    func testStatsEmptyReturnsZero() {
        let capture = freshCapture()
        let (perWord, total) = capture.stats()
        XCTAssertEqual(perWord.count, 0)
        XCTAssertEqual(total, 0)
    }

    func testStatsCountsCorrectly() {
        let capture = freshCapture()
        createTestImage(in: capture.baseDir, word: "laptop", count: 3)
        createTestImage(in: capture.baseDir, word: "cup", count: 5)

        let (perWord, total) = capture.stats()
        XCTAssertEqual(total, 8)
        XCTAssertEqual(perWord.count, 2)
        // Sorted alphabetically
        XCTAssertEqual(perWord[0].word, "cup")
        XCTAssertEqual(perWord[0].count, 5)
        XCTAssertEqual(perWord[1].word, "laptop")
        XCTAssertEqual(perWord[1].count, 3)
    }

    func testStatsHandlesUnderscoreWords() {
        let capture = freshCapture()
        createTestImage(in: capture.baseDir, word: "teddy bear", count: 2)

        let (perWord, _) = capture.stats()
        XCTAssertEqual(perWord.count, 1)
        XCTAssertEqual(perWord[0].word, "teddy bear")
        XCTAssertEqual(perWord[0].count, 2)
    }

    // MARK: - Export

    func testExportEmptyReturnsNil() {
        let capture = freshCapture()
        let url = capture.createExportArchive()
        XCTAssertNil(url)
    }

    func testExportCreatesZipFile() {
        let capture = freshCapture()
        createTestImage(in: capture.baseDir, word: "chair", count: 2)
        createTestImage(in: capture.baseDir, word: "table", count: 1)

        let url = capture.createExportArchive()
        XCTAssertNotNil(url)
        if let url = url {
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
            let data = try! Data(contentsOf: url)
            // ZIP files start with PK (0x504B)
            XCTAssertEqual(data[0], 0x50)
            XCTAssertEqual(data[1], 0x4B)
            // Clean up
            try? FileManager.default.removeItem(at: url)
        }
    }

    // MARK: - Clear

    func testClearRemovesAllData() {
        let capture = freshCapture()
        createTestImage(in: capture.baseDir, word: "phone", count: 4)
        XCTAssertEqual(capture.stats().total, 4)

        capture.clearAll()
        XCTAssertEqual(capture.stats().total, 0)
    }
}
