import XCTest
import Vision

/// Enumerates all VNClassifyImageRequest labels and checks mapping coverage.
/// This test reveals which VN categories we're missing in label_mappings.json.
final class VNTaxonomyTests: XCTestCase {

    func testDumpAllVNLabels() throws {
        // Get all known classification labels from Vision
        let allLabels = try VNClassifyImageRequest.knownClassifications(forRevision: VNClassifyImageRequestRevision1)
        let identifiers = allLabels.map(\.identifier).sorted()

        // Write to a file in the app's documents directory for easy retrieval
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let outputURL = docsDir.appendingPathComponent("vn_taxonomy.txt")
        let content = identifiers.joined(separator: "\n")
        try content.write(to: outputURL, atomically: true, encoding: .utf8)
        print("=== VN TAXONOMY: \(identifiers.count) labels written to \(outputURL.path) ===")

        // Also print in batches to avoid stdout truncation
        let batchSize = 100
        for batchStart in stride(from: 0, to: identifiers.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, identifiers.count)
            let batch = identifiers[batchStart..<batchEnd]
            print("=== BATCH \(batchStart)-\(batchEnd-1) ===")
            for id in batch {
                print("  \(id)")
            }
        }
        print("=== END VN TAXONOMY ===")

        XCTAssertGreaterThan(identifiers.count, 100, "Should have many classification labels")
    }

    func testMappingCoverage() throws {
        // Load label_mappings.json
        let bundle = Bundle(for: type(of: self))
        // The mappings are in the app bundle, try main bundle too
        let mappingsURL = bundle.url(forResource: "label_mappings", withExtension: "json")
            ?? Bundle.main.url(forResource: "label_mappings", withExtension: "json")

        guard let url = mappingsURL, let data = try? Data(contentsOf: url) else {
            // If can't load from test bundle, load from file path directly
            let projectPath = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("ARYA/Resources/label_mappings.json")
            let data = try Data(contentsOf: projectPath)
            let mappings = try JSONDecoder().decode([String: String?].self, from: data)
            try checkCoverage(mappings: mappings)
            return
        }
        let mappings = try JSONDecoder().decode([String: String?].self, from: data)
        try checkCoverage(mappings: mappings)
    }

    private func checkCoverage(mappings: [String: String?]) throws {
        let allLabels = try VNClassifyImageRequest.knownClassifications(forRevision: VNClassifyImageRequestRevision1)
        let identifiers = allLabels.map(\.identifier)

        var unmapped: [String] = []
        var mapped: [String: String] = [:]
        var nullMapped: [String] = []

        for id in identifiers {
            // Try exact match, then underscore→space, then lowercase
            let variations = [id, id.replacingOccurrences(of: "_", with: " "), id.lowercased().replacingOccurrences(of: "_", with: " ")]
            var found = false

            for variant in variations {
                if let value = mappings[variant] {
                    if let word = value {
                        mapped[id] = word
                    } else {
                        nullMapped.append(id)
                    }
                    found = true
                    break
                }
            }

            if !found {
                unmapped.append(id)
            }
        }

        print("=== MAPPING COVERAGE ===")
        print("Total VN labels: \(identifiers.count)")
        print("Mapped to vocabulary: \(mapped.count)")
        print("Null-mapped (skipped): \(nullMapped.count)")
        print("UNMAPPED (gaps): \(unmapped.count)")
        print("")
        print("=== UNMAPPED LABELS (these are INVISIBLE to the app) ===")
        for label in unmapped.sorted() {
            print("  \(label)")
        }
        print("")
        print("=== MAPPED LABELS ===")
        for (label, word) in mapped.sorted(by: { $0.key < $1.key }) {
            print("  \(label) → \(word)")
        }
        print("=== END COVERAGE ===")
    }

    /// Test classification on a simple test image to see what VN returns
    func testClassifyBlankImage() throws {
        // Create a simple solid-color image to see baseline behavior
        let size = CGSize(width: 224, height: 224)
        UIGraphicsBeginImageContext(size)
        UIColor.white.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let ciImage = CIImage(image: image)!
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])

        let request = VNClassifyImageRequest()
        try handler.perform([request])

        guard let results = request.results else {
            XCTFail("No results from VNClassifyImageRequest")
            return
        }

        // Show top 20 results for a blank white image
        let top20 = results.prefix(20)
        print("=== TOP 20 for BLANK WHITE IMAGE ===")
        for obs in top20 {
            print("  \(obs.identifier): \(String(format: "%.4f", obs.confidence))")
        }
        print("=== END ===")
    }
}
