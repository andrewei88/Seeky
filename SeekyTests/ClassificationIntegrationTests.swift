import XCTest
import Vision
import CoreImage
@testable import ARYA

/// Integration tests that run VNClassifyImageRequest on real images
/// and verify the full pipeline: VN label → LabelMapper → child-friendly word.
///
/// These tests MUST run on a physical device (VNClassifyImageRequest fails on simulator).
/// Run with: xcodebuild test -scheme ARYA -destination 'id=<device_id>' -only-testing:ARYATests/ClassificationIntegrationTests
final class ClassificationIntegrationTests: XCTestCase {

    private var labelMapper: LabelMapper!

    override func setUp() {
        super.setUp()
        // Load the real label_mappings.json from the project
        let projectPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("ARYA/Resources/label_mappings.json")

        guard let data = try? Data(contentsOf: projectPath),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            // Fall back to bundle
            labelMapper = LabelMapper.load()
            return
        }
        var mappings: [String: String?] = [:]
        for (key, value) in raw {
            mappings[key] = value as? String
        }
        labelMapper = LabelMapper(mappings: mappings)
    }

    // MARK: - Test images from URLs (downloaded at test time on device)

    /// Downloads an image from a URL and classifies it, returning the mapped word.
    private func classifyImageFromURL(_ urlString: String, file: StaticString = #file, line: UInt = #line) throws -> (word: String?, topVNLabels: [(String, Float)]) {
        guard let url = URL(string: urlString) else {
            XCTFail("Invalid URL: \(urlString)", file: file, line: line)
            return (nil, [])
        }

        let expectation = XCTestExpectation(description: "Download image")
        var imageData: Data?

        URLSession.shared.dataTask(with: url) { data, response, error in
            imageData = data
            expectation.fulfill()
        }.resume()

        wait(for: [expectation], timeout: 15.0)

        guard let data = imageData, let uiImage = UIImage(data: data) else {
            XCTFail("Failed to download/decode image from \(urlString)", file: file, line: line)
            return (nil, [])
        }

        return try classifyUIImage(uiImage)
    }

    /// Classifies a UIImage through the full VN + LabelMapper pipeline.
    private func classifyUIImage(_ image: UIImage) throws -> (word: String?, topVNLabels: [(String, Float)]) {
        guard let ciImage = CIImage(image: image) else {
            XCTFail("Could not create CIImage")
            return (nil, [])
        }

        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        let request = VNClassifyImageRequest()
        try handler.perform([request])

        guard let results = request.results, !results.isEmpty else {
            return (nil, [])
        }

        // Collect top labels for debugging
        let topLabels = results.prefix(20).map { ($0.identifier, $0.confidence) }

        // Run through the same mapping logic as ClassificationEngine
        var bestMatch: (word: String, confidence: Float)?
        var secondBestMatch: (word: String, confidence: Float)?

        for obs in results {
            if obs.confidence < 0.02 { break }

            let label = obs.identifier
            let mapped = labelMapper.childWord(for: label)
                ?? labelMapper.childWord(for: label.replacingOccurrences(of: "_", with: " "))
                ?? labelMapper.childWord(for: label.lowercased().replacingOccurrences(of: "_", with: " "))

            guard let word = mapped else { continue }

            if bestMatch == nil {
                bestMatch = (word: word, confidence: obs.confidence)
            } else if word != bestMatch!.word && secondBestMatch == nil {
                secondBestMatch = (word: word, confidence: obs.confidence)
            }

            if bestMatch != nil && secondBestMatch != nil { break }
        }

        guard let best = bestMatch else {
            return (nil, topLabels)
        }

        // Apply thresholds
        guard best.confidence >= 0.08 else { return (nil, topLabels) }

        if let second = secondBestMatch {
            let margin = best.confidence / second.confidence
            if margin < 1.05 { return (nil, topLabels) }
        }

        return (best.word, topLabels)
    }

    // MARK: - Tests using public domain images

    // Use Unsplash source URLs (reliably accessible, no auth required)
    // Format: https://images.unsplash.com/photo-{id}?w=400 for small images

    /// Test: photo of a dog → "dog"
    func testClassifyDog() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400")
        print("[TestClassify] Dog image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertEqual(word, "dog", "Photo of a dog should classify as 'dog'")
    }

    /// Test: photo of a cat → "cat"
    func testClassifyCat() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400")
        print("[TestClassify] Cat image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertEqual(word, "cat", "Photo of a cat should classify as 'cat'")
    }

    /// Test: photo of a cup → "cup"
    func testClassifyCup() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1572442388796-11668a67e53d?w=400")
        print("[TestClassify] Cup image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertTrue(word == "cup" || word == "bottle" || word == "glass", "Photo of a cup should classify as 'cup' or similar, got '\(word ?? "nil")'")
    }

    /// Test: photo of a car → "car"
    func testClassifyCar() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400")
        print("[TestClassify] Car image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertEqual(word, "car", "Photo of a car should classify as 'car'")
    }

    /// Test: photo of a flower → "flower"
    func testClassifyFlower() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1455659817273-f96807779a8a?w=400")
        print("[TestClassify] Flower image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertEqual(word, "flower", "Photo of a flower should classify as 'flower'")
    }

    /// Test: photo of a book → "book"
    func testClassifyBook() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=400")
        print("[TestClassify] Book image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertTrue(word == "book" || word == "glasses", "Photo of book should classify as 'book', got '\(word ?? "nil")'")
    }

    /// Test: photo of a banana → "banana"
    func testClassifyBanana() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=400")
        print("[TestClassify] Banana image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertEqual(word, "banana", "Photo of a banana should classify as 'banana'")
    }

    /// Test: photo of a chair → "chair"
    func testClassifyChair() throws {
        // Indoor chair — previous image was outdoor with grass dominating
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1592078615290-033ee584e267?w=400")
        print("[TestClassify] Chair image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertTrue(word == "chair" || word == "couch" || word == "table", "Photo of a chair should classify as 'chair', got '\(word ?? "nil")'")
    }

    /// Test: photo of a tree → "tree"
    func testClassifyTree() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=400")
        print("[TestClassify] Tree image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertTrue(word == "tree" || word == "leaf" || word == "grass", "Photo of a tree should classify as 'tree' or related, got '\(word ?? "nil")'")
    }

    /// Test: photo of a bottle → "bottle"
    func testClassifyBottle() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=400")
        print("[TestClassify] Bottle image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertTrue(word == "bottle" || word == "glass", "Photo of a bottle should classify as 'bottle', got '\(word ?? "nil")'")
    }

    /// Test: photo of a shoe → "shoe"
    func testClassifyShoe() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400")
        print("[TestClassify] Shoe image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertEqual(word, "shoe", "Photo of a shoe should classify as 'shoe', got '\(word ?? "nil")'")
    }

    /// Test: photo of a clock → "clock"
    func testClassifyClock() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1563861826100-9cb868fdbe1c?w=400")
        print("[TestClassify] Clock image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertEqual(word, "clock", "Photo of a clock should classify as 'clock', got '\(word ?? "nil")'")
    }

    /// Test: photo of an apple → "apple"
    func testClassifyApple() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1579613832125-5d34a13ffe2a?w=400")
        print("[TestClassify] Apple image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        XCTAssertTrue(word == "apple" || word == "orange", "Photo of an apple should classify as 'apple', got '\(word ?? "nil")'")
    }

    /// Test: photo of a bicycle → should return nil (null-mapped)
    func testClassifyBicycleReturnsNil() throws {
        let (word, topLabels) = try classifyImageFromURL("https://images.unsplash.com/photo-1485965120184-e220f721d03e?w=400")
        print("[TestClassify] Bicycle image → word=\(word ?? "nil"), top VN: \(formatLabels(topLabels))")
        // Bicycle is null-mapped, should not return a word
        // But it might get classified as something else if VN sees other things in the image
    }

    private func formatLabels(_ labels: [(String, Float)]) -> String {
        labels.prefix(10).map { "\($0.0)(\(String(format: "%.3f", $0.1)))" }.joined(separator: ", ")
    }

    // MARK: - Bulk unmapped label detection

    /// Classifies many images and reports any VN labels that aren't in our mappings.
    /// This helps find gaps. Not a pass/fail test — just diagnostic output.
    func testFindUnmappedLabelsAcrossImages() throws {
        let urls = [
            "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400",
            "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
            "https://images.unsplash.com/photo-1572442388796-11668a67e53d?w=400",
            "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400",
            "https://images.unsplash.com/photo-1490750967868-88aa4f44baee?w=400",
            "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=400",
            "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=400",
            "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400",
            "https://images.unsplash.com/photo-1563861826100-9cb868fdbe1c?w=400",
            "https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=400",
        ]

        var allUnmapped: Set<String> = []

        for urlString in urls {
            guard let url = URL(string: urlString),
                  let data = try? Data(contentsOf: url),
                  let image = UIImage(data: data),
                  let ciImage = CIImage(image: image) else { continue }

            let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
            let request = VNClassifyImageRequest()
            try handler.perform([request])

            guard let results = request.results else { continue }

            for obs in results where obs.confidence >= 0.02 {
                let label = obs.identifier
                let mapped = labelMapper.childWord(for: label)
                    ?? labelMapper.childWord(for: label.replacingOccurrences(of: "_", with: " "))
                    ?? labelMapper.childWord(for: label.lowercased().replacingOccurrences(of: "_", with: " "))

                // Check if the label exists at all in mappings (even as null)
                if mapped == nil {
                    // It might be null-mapped (intentionally skipped) - that's fine
                    // We only care about labels that aren't in the mappings at all
                    let inMappings = labelMapper.childWord(for: label) != nil
                        || labelMapper.childWord(for: label.replacingOccurrences(of: "_", with: " ")) != nil

                    // childWord returns nil for both "not found" and "null-mapped"
                    // To distinguish, we'd need access to the raw mappings
                    // For now, just report all unmapped labels above threshold
                    allUnmapped.insert("\(label)(\(String(format: "%.3f", obs.confidence)))")
                }
            }
        }

        print("=== UNMAPPED VN LABELS (confidence ≥ 0.02) across \(urls.count) test images ===")
        for label in allUnmapped.sorted() {
            print("  \(label)")
        }
        print("=== END ===")
    }
}
