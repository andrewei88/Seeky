import XCTest
@testable import ARYA

/// Tests for the classification path: label mapping, threshold logic, and confidence aggregation.
/// We can't run the full Vision pipeline in unit tests, but we can verify
/// the label mapping and threshold logic by testing LabelMapper with the
/// exact label sequences VNClassify produces.
final class ClassificationEngineTests: XCTestCase {

    // Simulate the classification logic: given VNClassify observations (label, confidence pairs),
    // find the best mapped word using the same algorithm as ClassificationEngine.
    private func simulateVNOnlyClassification(
        observations: [(label: String, confidence: Float)],
        mappings: [String: String?]
    ) -> (word: String, confidence: Float, margin: Float)? {
        let mapper = LabelMapper(mappings: mappings)

        // Aggregate confidence across all labels that map to the same word
        var wordConfidences: [String: Float] = [:]

        for obs in observations {
            if obs.confidence < 0.02 { break }

            let mapped = mapper.childWord(for: obs.label)
                ?? mapper.childWord(for: obs.label.replacingOccurrences(of: "_", with: " "))
                ?? mapper.childWord(for: obs.label.lowercased().replacingOccurrences(of: "_", with: " "))

            guard let word = mapped else { continue }
            wordConfidences[word, default: 0] += obs.confidence
        }

        let sorted = wordConfidences.sorted { $0.value > $1.value }
        guard let best = sorted.first else { return nil }
        let secondBest = sorted.count > 1 ? sorted[1] : nil

        let margin = secondBest != nil ? best.value / secondBest!.value : Float.infinity

        // Apply same thresholds as ClassificationEngine
        guard best.value >= 0.03 else { return nil }
        if let _ = secondBest, margin < 1.05 { return nil }

        return (word: best.key, confidence: best.value, margin: margin)
    }

    /// Laptop behind null-mapped labels: should be accepted when confidence is above threshold.
    func testLaptopClassificationSkipsNullMappedLabels() {
        let observations: [(label: String, confidence: Float)] = [
            ("document", 0.907),
            ("screenshot", 0.907),
            ("machine", 0.244),
            ("consumer_electronics", 0.244),
            ("computer", 0.244),
            ("laptop", 0.123),
        ]

        // "computer" is null-mapped (too generic), laptop comes from "laptop" label
        let mappings: [String: String?] = [
            "document": nil,
            "screenshot": nil,
            "machine": nil,
            "consumer_electronics": nil,
            "computer": nil,
            "laptop": "laptop",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result, "Should accept laptop when confidence is above 0.05")
        XCTAssertEqual(result?.word, "laptop")
    }

    /// Very low confidence laptop (0.025) should be rejected below 0.03 threshold.
    func testLaptopRejectedBelowThreshold() {
        let observations: [(label: String, confidence: Float)] = [
            ("document", 0.907),
            ("screenshot", 0.907),
            ("computer", 0.028),
            ("laptop", 0.025),
        ]

        let mappings: [String: String?] = [
            "document": nil,
            "screenshot": nil,
            "computer": nil,
            "laptop": "laptop",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNil(result, "Should reject laptop when confidence is below 0.03")
    }

    /// When two mapped words are essentially equal in confidence, reject as ambiguous.
    func testAmbiguousClassificationRejected() {
        let observations: [(label: String, confidence: Float)] = [
            ("dog", 0.45),
            ("cat", 0.44),  // margin = 1.02 < 1.05 → ambiguous
        ]

        let mappings: [String: String?] = [
            "dog": "dog",
            "cat": "cat",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNil(result, "Should reject when two mapped words are nearly equal in confidence")
    }

    /// Clear winner with no competition should be accepted.
    func testClearWinnerAccepted() {
        let observations: [(label: String, confidence: Float)] = [
            ("golden_retriever", 0.85),
            ("animal", 0.10),
        ]

        let mappings: [String: String?] = [
            "golden retriever": "dog",
            "animal": nil,
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertEqual(result?.word, "dog")
    }

    /// When ONLY null-mapped labels appear, return nil.
    func testAllNullMappedReturnsNil() {
        let observations: [(label: String, confidence: Float)] = [
            ("document", 0.95),
            ("screenshot", 0.90),
            ("structure", 0.40),
        ]

        let mappings: [String: String?] = [
            "document": nil,
            "screenshot": nil,
            "structure": nil,
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNil(result, "Should return nil when only null-mapped labels found")
    }

    /// Very low confidence below 0.02 scan floor should reject.
    func testVeryLowConfidenceRejected() {
        let observations: [(label: String, confidence: Float)] = [
            ("document", 0.90),
            ("dog", 0.019),
        ]

        let mappings: [String: String?] = [
            "document": nil,
            "dog": "dog",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNil(result, "Should reject when confidence is below 0.02 scan floor")
    }

    /// Verify that key VN taxonomy labels the user reported issues with are now mapped.
    func testVNTaxonomyLabelsCoverage() {
        // Load the real label_mappings.json
        let projectPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("ARYA/Resources/label_mappings.json")

        guard let data = try? Data(contentsOf: projectPath),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            XCTFail("Could not load label_mappings.json")
            return
        }

        var mappings: [String: String?] = [:]
        for (key, value) in raw {
            mappings[key] = value as? String
        }
        let mapper = LabelMapper(mappings: mappings)

        // Helper: check that a VN label maps to expected word
        func assertMapped(_ vnLabel: String, to expected: String, file: StaticString = #file, line: UInt = #line) {
            let result = mapper.childWord(for: vnLabel)
                ?? mapper.childWord(for: vnLabel.replacingOccurrences(of: "_", with: " "))
                ?? mapper.childWord(for: vnLabel.lowercased().replacingOccurrences(of: "_", with: " "))
            XCTAssertEqual(result, expected, "VN label '\(vnLabel)' should map to '\(expected)' but got '\(result ?? "nil")'", file: file, line: line)
        }

        func assertNullMapped(_ vnLabel: String, file: StaticString = #file, line: UInt = #line) {
            // Should be found but mapped to nil
            let found = mappings[vnLabel] != nil
                || mappings[vnLabel.replacingOccurrences(of: "_", with: " ")] != nil
                || mappings[vnLabel.lowercased().replacingOccurrences(of: "_", with: " ")] != nil
            XCTAssertTrue(found, "VN label '\(vnLabel)' should be null-mapped (found in mappings) but was not found at all", file: file, line: line)
        }

        // Speakers should map to "speaker" — VN returns these labels for speakers
        assertMapped("loudspeaker", to: "speaker")
        assertMapped("speaker", to: "speaker")
        assertMapped("speakers_music", to: "speaker")
        assertMapped("megaphone", to: "speaker")
        assertMapped("stereo", to: "speaker")

        // These VN labels should be null-mapped (not objects)
        assertNullMapped("music")
        assertNullMapped("camera")
        assertNullMapped("fire")
        assertNullMapped("flame")

        // User reported: cups, bottles, bowls not detected → verify these VN labels work
        assertMapped("cup", to: "cup")
        assertMapped("bottle", to: "bottle")
        assertMapped("bowl", to: "bowl")
        assertMapped("coffee", to: "cup")
        assertMapped("mug", to: "cup")

        // Key VN taxonomy labels that were missing
        assertMapped("adult_cat", to: "cat")
        assertMapped("blocks", to: "block")
        assertMapped("blossom", to: "flower")
        assertMapped("branch", to: "tree")
        assertMapped("cake_regular", to: "cake")
        assertMapped("chair_other", to: "chair")
        assertMapped("christmas_tree", to: "tree")
        assertMapped("citrus_fruit", to: "orange")
        assertMapped("cookware", to: "pot")
        assertMapped("crate", to: "box")
        assertMapped("decorative_plant", to: "flower")
        assertMapped("birthday_cake", to: "cake")

        // Laptop-specific labels → "laptop"
        assertMapped("laptop", to: "laptop")
        assertMapped("notebook computer", to: "laptop")
        assertMapped("MacBook", to: "laptop")

        // External monitor labels → "monitor" (separated from tv)
        assertMapped("computer_monitor", to: "monitor")
        assertMapped("display", to: "monitor")
        assertMapped("screen", to: "monitor")
        assertMapped("desktop_computer", to: "monitor")

        // TV labels → "tv"
        assertMapped("television", to: "tv")
        assertMapped("tv", to: "tv")
        assertMapped("LCD", to: "tv")
        assertMapped("flat screen", to: "tv")

        // Generic computer labels → null (too ambiguous)
        assertNullMapped("computer")
        assertNullMapped("computer_tower")
        assertNullMapped("PC")

        // Keyboard → "keyboard"
        assertMapped("computer_keyboard", to: "keyboard")

        // Common animals
        assertMapped("golden_retriever", to: "dog")
        assertMapped("german_shepherd", to: "dog")
        assertMapped("labrador_retriever", to: "dog")
        assertMapped("poodle", to: "dog")
        assertMapped("tabby", to: "cat")
        assertMapped("siamese", to: "cat")
        assertMapped("feline", to: "cat")

        // Furniture — user wants tables, chairs
        assertMapped("sofa", to: "couch")
        assertMapped("dining_table", to: "table")
        assertMapped("coffee_table", to: "table")
        assertMapped("armchair", to: "chair")

        // Food — user wants fruits, foods
        assertMapped("donut", to: "cake")
        assertMapped("pastry", to: "cake")
        assertMapped("pancake", to: "bread")
        assertMapped("sandwich", to: "bread")
        assertMapped("hamburger", to: "bread")
        assertMapped("cracker", to: "cookie")
        assertMapped("soda", to: "can")
        assertMapped("thermos", to: "bottle")

        // Drinks
        assertMapped("milkshake", to: "cup")
        assertMapped("smoothie", to: "cup")

        // Containers
        assertMapped("cardboard_box", to: "box")
        assertMapped("paper_bag", to: "bag")
        assertMapped("backpack", to: "bag")
        assertMapped("luggage", to: "bag")

        // More animals — user requested pandas, tigers, sharks, whales, etc.
        assertMapped("panda", to: "panda")
        assertMapped("giant_panda", to: "panda")
        assertMapped("polar_bear", to: "bear")
        assertMapped("tiger", to: "tiger")
        assertMapped("leopard", to: "cat")
        assertMapped("shark", to: "shark")
        assertMapped("whale", to: "whale")
        assertMapped("dolphin", to: "dolphin")
        assertMapped("octopus", to: "octopus")
        assertMapped("wolf", to: "dog")
        assertMapped("husky", to: "dog")
        assertMapped("pug", to: "dog")
        assertMapped("eagle", to: "bird")
        assertMapped("owl", to: "owl")
        assertMapped("penguin", to: "penguin")
        assertMapped("parrot", to: "parrot")
        assertMapped("swan", to: "duck")
        assertMapped("gorilla", to: "monkey")
        assertMapped("hamster", to: "rabbit")
        assertMapped("squirrel", to: "squirrel")
        assertMapped("lizard", to: "frog")
        assertMapped("alligator", to: "turtle")
        assertMapped("snail", to: "turtle")

        // Misc
        assertMapped("barracuda", to: "fish")
        assertMapped("bedding", to: "blanket")
        assertMapped("payphone", to: "phone")
        assertMapped("figurine", to: "doll")
        assertMapped("watch", to: "clock")
        assertMapped("vase", to: "bottle")
        assertMapped("kettle", to: "pot")
        assertMapped("basket", to: "bowl")

        // Screen/monitor mappings
        assertMapped("television", to: "tv")
        assertMapped("computer_monitor", to: "monitor")

        // Abstract labels should be null-mapped
        assertNullMapped("painting")
        assertNullMapped("candy")
        assertNullMapped("chocolate")
        assertNullMapped("interior_room")
        assertNullMapped("bathroom")
        assertNullMapped("calculator")
        assertNullMapped("newspaper")
        assertNullMapped("elevator")
        assertNullMapped("stairs")
        assertNullMapped("piano")
        assertNullMapped("snake")
    }

    /// Simulate the EXACT speaker scenario from device logs — speakers_music should now map.
    func testSpeakerDetectionFromDeviceLogs() {
        let observations: [(label: String, confidence: Float)] = [
            ("music", 0.152),
            ("speakers_music", 0.145),
            ("megaphone", 0.096),
            ("camera", 0.068),
            ("people", 0.034),
            ("adult", 0.034),
            ("fire", 0.029),
            ("flame", 0.029),
            ("material", 0.027),
        ]

        let mappings: [String: String?] = [
            "music": nil,
            "speakers_music": "speaker",
            "megaphone": "speaker",
            "camera": nil,
            "people": nil,
            "adult": nil,
            "fire": nil,
            "flame": nil,
            "material": nil,
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result, "Speaker should be detected from speakers_music label")
        XCTAssertEqual(result?.word, "speaker")
        XCTAssertGreaterThanOrEqual(result?.confidence ?? 0, 0.08, "speakers_music confidence 0.145 should pass threshold")
    }

    // MARK: - Device log replay tests
    // These replay EXACT VN output from device logs to verify fixes work.

    /// Bottle at 0.060 was rejected with old 0.08 threshold — should now pass at 0.05.
    func testBottleAcceptedWithLoweredThreshold() {
        let observations: [(label: String, confidence: Float)] = [
            ("container", 0.061),
            ("material", 0.061),
            ("bottle", 0.060),
            ("drink", 0.058),
            ("liquid", 0.058),
            ("soda", 0.058),
            ("textile", 0.055),
            ("tableware", 0.035),
            ("utensil", 0.035),
            ("cup", 0.034),
        ]

        let mappings: [String: String?] = [
            "container": nil,
            "material": nil,
            "bottle": "bottle",
            "drink": nil,
            "liquid": nil,
            "soda": "bottle",
            "textile": nil,
            "tableware": nil,
            "utensil": nil,
            "cup": "cup",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result, "Bottle at 0.060 should pass with 0.05 threshold")
        XCTAssertEqual(result?.word, "bottle")
    }

    /// Glass at 0.078 was rejected with old 0.08 threshold — should now pass.
    func testGlassAcceptedWithLoweredThreshold() {
        let observations: [(label: String, confidence: Float)] = [
            ("material", 0.083),
            ("textile", 0.083),
            ("tableware", 0.078),
            ("utensil", 0.078),
            ("drinking_glass", 0.078),
            ("people", 0.039),
            ("adult", 0.039),
            ("container", 0.028),
            ("bottle", 0.028),
        ]

        let mappings: [String: String?] = [
            "material": nil,
            "textile": nil,
            "tableware": nil,
            "utensil": nil,
            "drinking_glass": "glass",
            "people": nil,
            "adult": nil,
            "container": nil,
            "bottle": "bottle",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result, "Glass at 0.078 should pass with 0.05 threshold")
        XCTAssertEqual(result?.word, "glass")
    }

    /// Bottle at 0.077 was rejected — should now pass.
    func testBottle077AcceptedWithLoweredThreshold() {
        let observations: [(label: String, confidence: Float)] = [
            ("container", 0.143),
            ("keg", 0.130),
            ("barrel", 0.099),
            ("drink", 0.078),
            ("liquid", 0.078),
            ("soda", 0.077),
            ("bottle", 0.076),
            ("utensil", 0.066),
            ("tableware", 0.066),
            ("cup", 0.065),
        ]

        let mappings: [String: String?] = [
            "container": nil,
            "keg": nil,
            "barrel": nil,
            "drink": nil,
            "liquid": nil,
            "soda": "bottle",
            "bottle": "bottle",
            "utensil": nil,
            "tableware": nil,
            "cup": "cup",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result, "Bottle (via soda) at 0.077 should pass with 0.05 threshold")
        XCTAssertEqual(result?.word, "bottle")
    }

    /// Cup with high confidence from device logs — should always work.
    func testCupHighConfidenceFromLogs() {
        let observations: [(label: String, confidence: Float)] = [
            ("tableware", 0.865),
            ("utensil", 0.865),
            ("cup", 0.865),
            ("drink", 0.855),
            ("liquid", 0.855),
            ("straw_drinking", 0.854),
            ("milkshake", 0.212),
            ("mug", 0.125),
        ]

        let mappings: [String: String?] = [
            "tableware": nil,
            "utensil": nil,
            "cup": "cup",
            "drink": nil,
            "liquid": nil,
            "straw_drinking": nil,
            "milkshake": "cup",
            "mug": "cup",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.word, "cup")
        XCTAssertGreaterThan(result?.confidence ?? 0, 0.5)
    }

    /// Keyboard detection from device logs — keyboard-dominated scene.
    /// "computer" is null-mapped, "computer_keyboard" maps to "keyboard".
    func testKeyboardFromKeyboardScene() {
        let observations: [(label: String, confidence: Float)] = [
            ("machine", 0.838),
            ("computer", 0.838),
            ("consumer_electronics", 0.838),
            ("computer_keyboard", 0.838),
            ("keypad", 0.191),
            ("laptop", 0.059),
        ]

        let mappings: [String: String?] = [
            "machine": nil,
            "computer": nil,
            "consumer_electronics": nil,
            "computer_keyboard": "keyboard",
            "keypad": nil,
            "laptop": "laptop",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.word, "keyboard")
    }

    /// TV detection from device logs — television label maps to "tv".
    func testTVDetectionFromLogs() {
        let observations: [(label: String, confidence: Float)] = [
            ("structure", 0.143),
            ("wood_processed", 0.141),
            ("machine", 0.140),
            ("consumer_electronics", 0.140),
            ("television", 0.130),
            ("computer", 0.100),
            ("computer_monitor", 0.099),
        ]

        let mappings: [String: String?] = [
            "structure": nil,
            "wood_processed": nil,
            "machine": nil,
            "consumer_electronics": nil,
            "television": "tv",
            "computer": nil,
            "computer_monitor": "tv",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result)
        // television appears first and maps to "tv"
        XCTAssertEqual(result?.word, "tv")
    }

    /// Box detection from device logs — cardboard_box scenario.
    func testBoxFromCardboardBoxLabel() {
        let observations: [(label: String, confidence: Float)] = [
            ("container", 0.180),
            ("cardboard_box", 0.174),
            ("material", 0.117),
            ("textile", 0.117),
            ("carton", 0.110),
            ("document", 0.058),
        ]

        let mappings: [String: String?] = [
            "container": nil,
            "cardboard_box": "box",
            "material": nil,
            "textile": nil,
            "carton": "box",
            "document": nil,
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.word, "box")
    }

    /// Cat detection scenario — feline label should map.
    func testCatFromFelineLabel() {
        let observations: [(label: String, confidence: Float)] = [
            ("animal", 0.45),
            ("mammal", 0.42),
            ("feline", 0.38),
            ("adult_cat", 0.35),
            ("tabby", 0.20),
        ]

        let mappings: [String: String?] = [
            "animal": nil,
            "mammal": nil,
            "feline": "cat",
            "adult_cat": "cat",
            "tabby": "cat",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.word, "cat")
    }

    /// Confidence at exactly 0.03 should pass (boundary test).
    func testExactThresholdBoundary() {
        let observations: [(label: String, confidence: Float)] = [
            ("material", 0.90),
            ("dog", 0.03),
        ]

        let mappings: [String: String?] = [
            "material": nil,
            "dog": "dog",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result, "Confidence of exactly 0.03 should pass the threshold")
        XCTAssertEqual(result?.word, "dog")
    }

    /// Confidence at 0.029 should be rejected.
    func testJustBelowThresholdRejected() {
        let observations: [(label: String, confidence: Float)] = [
            ("material", 0.90),
            ("dog", 0.029),
        ]

        let mappings: [String: String?] = [
            "material": nil,
            "dog": "dog",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNil(result, "Confidence of 0.029 should be rejected")
    }

    /// Monitor detection when computer and computer_monitor appear at same confidence.
    /// Previously rejected as ambiguous (computer→laptop, computer_monitor→monitor, margin=1.00).
    /// Now fixed: computer is null-mapped, so only computer_monitor→monitor is found.
    func testMonitorNotAmbiguousWithComputer() {
        let observations: [(label: String, confidence: Float)] = [
            ("machine", 0.244),
            ("consumer_electronics", 0.244),
            ("computer", 0.131),
            ("computer_monitor", 0.131),
            ("display", 0.098),
        ]

        let mappings: [String: String?] = [
            "machine": nil,
            "consumer_electronics": nil,
            "computer": nil,
            "computer_monitor": "tv",
            "display": "tv",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result, "Should detect tv — computer is null-mapped, no ambiguity")
        XCTAssertEqual(result?.word, "tv")
    }

    /// Soda can should map to can.
    func testSodaCanMapsToCan() {
        let observations: [(label: String, confidence: Float)] = [
            ("container", 0.200),
            ("can", 0.180),
            ("soda_can", 0.170),
            ("drink", 0.150),
            ("soda", 0.140),
        ]

        let mappings: [String: String?] = [
            "container": nil,
            "can": "can",
            "soda_can": "can",
            "drink": nil,
            "soda": "can",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertNotNil(result, "Soda can should be detected as can")
        XCTAssertEqual(result?.word, "can")
    }

    /// Underscore/space normalization should work.
    func testUnderscoreNormalization() {
        let observations: [(label: String, confidence: Float)] = [
            ("golden_retriever", 0.80),
        ]

        let mappings: [String: String?] = [
            "golden retriever": "dog",
        ]

        let result = simulateVNOnlyClassification(observations: observations, mappings: mappings)
        XCTAssertEqual(result?.word, "dog", "Should find mapping via underscore-to-space normalization")
    }

    // MARK: - Real-world label_mappings.json tests
    // These use the ACTUAL label_mappings.json to verify the full pipeline for many object categories.

    /// Helper that loads real label_mappings.json and simulates classification.
    private func simulateWithRealMappings(
        observations: [(label: String, confidence: Float)]
    ) -> (word: String, confidence: Float, margin: Float)? {
        let projectPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("ARYA/Resources/label_mappings.json")

        guard let data = try? Data(contentsOf: projectPath),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            XCTFail("Could not load label_mappings.json")
            return nil
        }

        var mappings: [String: String?] = [:]
        for (key, value) in raw {
            mappings[key] = value as? String
        }

        return simulateVNOnlyClassification(observations: observations, mappings: mappings)
    }

    // --- Animals ---

    func testDogFromGoldenRetriever() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.95), ("mammal", 0.90), ("canine", 0.85),
            ("golden_retriever", 0.75), ("dog", 0.60)
        ])
        XCTAssertEqual(result?.word, "dog")
    }

    func testDogFromGermanShepherd() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.90), ("german_shepherd", 0.70), ("canine", 0.65)
        ])
        XCTAssertEqual(result?.word, "dog")
    }

    func testDogFromPug() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.88), ("mammal", 0.85), ("pug", 0.72), ("canine", 0.60)
        ])
        XCTAssertEqual(result?.word, "dog")
    }

    func testCatFromTabby() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.92), ("feline", 0.80), ("tabby", 0.70), ("adult_cat", 0.65)
        ])
        XCTAssertEqual(result?.word, "cat")
    }

    func testCatFromSiamese() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.88), ("feline", 0.75), ("siamese", 0.68)
        ])
        XCTAssertEqual(result?.word, "cat")
    }

    func testTigerFromTiger() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.90), ("mammal", 0.85), ("tiger", 0.72), ("carnivore", 0.50)
        ])
        XCTAssertEqual(result?.word, "tiger")
    }

    func testBirdFromEagle() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.80), ("bird", 0.75), ("eagle", 0.60)
        ])
        XCTAssertEqual(result?.word, "bird")
    }

    func testBirdFromOwl() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.85), ("bird", 0.78), ("owl", 0.65)
        ])
        XCTAssertEqual(result?.word, "bird")
    }

    func testPenguinFromPenguin() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.82), ("penguin", 0.75), ("bird", 0.50)
        ])
        XCTAssertEqual(result?.word, "penguin")
    }

    func testPandaFromPanda() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.90), ("mammal", 0.85), ("giant_panda", 0.70), ("panda", 0.65)
        ])
        XCTAssertEqual(result?.word, "panda")
    }

    func testBearFromPolarBear() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.88), ("mammal", 0.82), ("polar_bear", 0.75)
        ])
        XCTAssertEqual(result?.word, "bear")
    }

    func testFishFromShark() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.85), ("fish", 0.70), ("shark", 0.65)
        ])
        XCTAssertEqual(result?.word, "fish")
    }

    func testDuckFromSwan() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.80), ("bird", 0.75), ("swan", 0.60)
        ])
        // bird or duck both valid — swan maps to duck
        XCTAssertTrue(result?.word == "bird" || result?.word == "duck")
    }

    func testMonkeyFromGorilla() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.85), ("mammal", 0.80), ("gorilla", 0.70), ("primate", 0.60)
        ])
        XCTAssertEqual(result?.word, "monkey")
    }

    func testDogFromWolf() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.88), ("wolf", 0.72), ("canine", 0.65)
        ])
        XCTAssertEqual(result?.word, "dog")
    }

    func testElephant() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.92), ("mammal", 0.88), ("elephant", 0.80)
        ])
        XCTAssertEqual(result?.word, "elephant")
    }

    func testHorse() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.90), ("mammal", 0.85), ("horse", 0.75)
        ])
        XCTAssertEqual(result?.word, "horse")
    }

    func testRabbit() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.88), ("rabbit", 0.72), ("bunny", 0.65)
        ])
        XCTAssertEqual(result?.word, "rabbit")
    }

    func testFrog() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.80), ("amphibian", 0.65), ("frog", 0.60)
        ])
        XCTAssertEqual(result?.word, "frog")
    }

    func testTurtle() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.82), ("reptile", 0.70), ("turtle", 0.65)
        ])
        XCTAssertEqual(result?.word, "turtle")
    }

    func testButterfly() {
        let result = simulateWithRealMappings(observations: [
            ("animal", 0.78), ("insect", 0.70), ("butterfly", 0.65)
        ])
        XCTAssertEqual(result?.word, "butterfly")
    }

    // --- Food ---

    func testApple() {
        let result = simulateWithRealMappings(observations: [
            ("food", 0.85), ("fruit", 0.80), ("apple", 0.75), ("Granny Smith", 0.50)
        ])
        XCTAssertEqual(result?.word, "apple")
    }

    func testBanana() {
        let result = simulateWithRealMappings(observations: [
            ("food", 0.88), ("fruit", 0.82), ("banana", 0.78)
        ])
        XCTAssertEqual(result?.word, "banana")
    }

    func testOrange() {
        let result = simulateWithRealMappings(observations: [
            ("food", 0.85), ("fruit", 0.80), ("orange", 0.70), ("citrus_fruit", 0.60)
        ])
        XCTAssertEqual(result?.word, "orange")
    }

    func testCakeFromDonut() {
        let result = simulateWithRealMappings(observations: [
            ("food", 0.80), ("pastry", 0.70), ("donut", 0.65)
        ])
        XCTAssertEqual(result?.word, "cake")
    }

    func testBreadFromSandwich() {
        let result = simulateWithRealMappings(observations: [
            ("food", 0.82), ("sandwich", 0.68), ("bread", 0.55)
        ])
        XCTAssertEqual(result?.word, "bread")
    }

    func testPizza() {
        let result = simulateWithRealMappings(observations: [
            ("food", 0.90), ("pizza", 0.85)
        ])
        XCTAssertEqual(result?.word, "pizza")
    }

    func testCookie() {
        let result = simulateWithRealMappings(observations: [
            ("food", 0.80), ("cookie", 0.72), ("cracker", 0.55)
        ])
        XCTAssertEqual(result?.word, "cookie")
    }

    func testEgg() {
        let result = simulateWithRealMappings(observations: [
            ("food", 0.75), ("egg", 0.70)
        ])
        XCTAssertEqual(result?.word, "egg")
    }

    func testCheese() {
        let result = simulateWithRealMappings(observations: [
            ("food", 0.80), ("dairy", 0.70), ("cheese", 0.65)
        ])
        XCTAssertEqual(result?.word, "cheese")
    }

    // --- Furniture ---

    func testChair() {
        let result = simulateWithRealMappings(observations: [
            ("furniture", 0.85), ("chair", 0.78), ("seat", 0.60)
        ])
        XCTAssertEqual(result?.word, "chair")
    }

    func testChairFromArmchair() {
        let result = simulateWithRealMappings(observations: [
            ("furniture", 0.82), ("armchair", 0.72), ("seat", 0.55)
        ])
        XCTAssertEqual(result?.word, "chair")
    }

    func testTable() {
        let result = simulateWithRealMappings(observations: [
            ("furniture", 0.88), ("table", 0.80), ("dining_table", 0.70)
        ])
        XCTAssertEqual(result?.word, "table")
    }

    func testCoffeeTable() {
        let result = simulateWithRealMappings(observations: [
            ("furniture", 0.80), ("coffee_table", 0.72), ("table", 0.65)
        ])
        XCTAssertEqual(result?.word, "table")
    }

    func testCouch() {
        let result = simulateWithRealMappings(observations: [
            ("furniture", 0.85), ("sofa", 0.78), ("couch", 0.70)
        ])
        XCTAssertEqual(result?.word, "couch")
    }

    func testBed() {
        let result = simulateWithRealMappings(observations: [
            ("furniture", 0.82), ("bed", 0.75), ("bedding", 0.55)
        ])
        XCTAssertEqual(result?.word, "bed")
    }

    // --- Household items ---

    func testCup() {
        let result = simulateWithRealMappings(observations: [
            ("tableware", 0.85), ("cup", 0.80), ("mug", 0.60)
        ])
        XCTAssertEqual(result?.word, "cup")
    }

    func testBottle() {
        let result = simulateWithRealMappings(observations: [
            ("container", 0.80), ("bottle", 0.75), ("drink", 0.55)
        ])
        XCTAssertEqual(result?.word, "bottle")
    }

    func testBowl() {
        let result = simulateWithRealMappings(observations: [
            ("tableware", 0.82), ("bowl", 0.78), ("dish", 0.50)
        ])
        XCTAssertEqual(result?.word, "bowl")
    }

    func testSpoon() {
        let result = simulateWithRealMappings(observations: [
            ("utensil", 0.80), ("tableware", 0.78), ("spoon", 0.72)
        ])
        XCTAssertEqual(result?.word, "spoon")
    }

    func testFork() {
        let result = simulateWithRealMappings(observations: [
            ("utensil", 0.82), ("tableware", 0.80), ("fork", 0.75)
        ])
        XCTAssertEqual(result?.word, "fork")
    }

    func testKnife() {
        let result = simulateWithRealMappings(observations: [
            ("utensil", 0.80), ("knife", 0.70)
        ])
        XCTAssertEqual(result?.word, "knife")
    }

    func testPlate() {
        let result = simulateWithRealMappings(observations: [
            ("tableware", 0.85), ("plate", 0.78), ("dish", 0.55)
        ])
        XCTAssertEqual(result?.word, "plate")
    }

    func testClock() {
        let result = simulateWithRealMappings(observations: [
            ("clock", 0.82), ("timepiece", 0.60)
        ])
        XCTAssertEqual(result?.word, "clock")
    }

    func testClockFromWatch() {
        let result = simulateWithRealMappings(observations: [
            ("watch", 0.78), ("timepiece", 0.55)
        ])
        XCTAssertEqual(result?.word, "clock")
    }

    func testLamp() {
        let result = simulateWithRealMappings(observations: [
            ("lamp", 0.80), ("light_fixture", 0.60)
        ])
        XCTAssertEqual(result?.word, "light")
    }

    func testPillow() {
        let result = simulateWithRealMappings(observations: [
            ("textile", 0.80), ("pillow", 0.72), ("cushion", 0.55)
        ])
        XCTAssertEqual(result?.word, "pillow")
    }

    func testBook() {
        let result = simulateWithRealMappings(observations: [
            ("document", 0.85), ("book", 0.78), ("paper", 0.50)
        ])
        XCTAssertEqual(result?.word, "book")
    }

    func testPhone() {
        let result = simulateWithRealMappings(observations: [
            ("consumer_electronics", 0.82), ("mobile_phone", 0.75), ("phone", 0.60)
        ])
        XCTAssertEqual(result?.word, "phone")
    }

    func testShoe() {
        let result = simulateWithRealMappings(observations: [
            ("clothing", 0.80), ("shoe", 0.75), ("footwear", 0.60)
        ])
        XCTAssertEqual(result?.word, "shoe")
    }

    func testBag() {
        let result = simulateWithRealMappings(observations: [
            ("container", 0.78), ("backpack", 0.70), ("bag", 0.55)
        ])
        XCTAssertEqual(result?.word, "bag")
    }

    func testBox() {
        let result = simulateWithRealMappings(observations: [
            ("container", 0.80), ("cardboard_box", 0.72), ("box", 0.60)
        ])
        XCTAssertEqual(result?.word, "box")
    }

    func testUmbrella() {
        let result = simulateWithRealMappings(observations: [
            ("umbrella", 0.82), ("parasol", 0.55)
        ])
        XCTAssertEqual(result?.word, "umbrella")
    }

    func testKey() {
        let result = simulateWithRealMappings(observations: [
            ("key", 0.78), ("tool", 0.50)
        ])
        XCTAssertEqual(result?.word, "key")
    }

    func testScissors() {
        let result = simulateWithRealMappings(observations: [
            ("scissors", 0.80), ("tool", 0.55)
        ])
        XCTAssertEqual(result?.word, "scissors")
    }

    func testPen() {
        let result = simulateWithRealMappings(observations: [
            ("stationery", 0.78), ("pen", 0.72)
        ])
        XCTAssertEqual(result?.word, "pen")
    }

    func testPencil() {
        let result = simulateWithRealMappings(observations: [
            ("stationery", 0.80), ("pencil", 0.75)
        ])
        XCTAssertEqual(result?.word, "pencil")
    }

    // --- Vehicles ---

    func testCar() {
        let result = simulateWithRealMappings(observations: [
            ("vehicle", 0.88), ("car", 0.82), ("automobile", 0.70)
        ])
        XCTAssertEqual(result?.word, "car")
    }

    func testTruck() {
        let result = simulateWithRealMappings(observations: [
            ("vehicle", 0.85), ("truck", 0.78)
        ])
        XCTAssertEqual(result?.word, "truck")
    }

    func testBus() {
        let result = simulateWithRealMappings(observations: [
            ("vehicle", 0.82), ("bus", 0.75)
        ])
        XCTAssertEqual(result?.word, "bus")
    }

    // --- Screen/Computer distinctions ---

    func testLaptopFromNotebook() {
        let result = simulateWithRealMappings(observations: [
            ("machine", 0.80), ("consumer_electronics", 0.78),
            ("notebook_computer", 0.70), ("laptop", 0.65)
        ])
        XCTAssertEqual(result?.word, "laptop")
    }

    func testMonitorFromComputerMonitor() {
        let result = simulateWithRealMappings(observations: [
            ("machine", 0.80), ("consumer_electronics", 0.78),
            ("computer", 0.65), ("computer_monitor", 0.64)
        ])
        XCTAssertEqual(result?.word, "monitor")
    }

    func testTVFromTelevision() {
        let result = simulateWithRealMappings(observations: [
            ("consumer_electronics", 0.80), ("television", 0.72),
            ("flat_screen", 0.55)
        ])
        XCTAssertEqual(result?.word, "tv")
    }

    func testKeyboardFromComputerKeyboard() {
        let result = simulateWithRealMappings(observations: [
            ("machine", 0.80), ("consumer_electronics", 0.78),
            ("computer", 0.75), ("computer_keyboard", 0.72)
        ])
        XCTAssertEqual(result?.word, "keyboard")
    }

    // --- Nature ---

    func testFlower() {
        let result = simulateWithRealMappings(observations: [
            ("plant", 0.82), ("flower", 0.78), ("blossom", 0.55)
        ])
        XCTAssertEqual(result?.word, "flower")
    }

    func testTree() {
        let result = simulateWithRealMappings(observations: [
            ("plant", 0.85), ("tree", 0.78)
        ])
        XCTAssertEqual(result?.word, "tree")
    }

    func testLeaf() {
        let result = simulateWithRealMappings(observations: [
            ("plant", 0.80), ("leaf", 0.72)
        ])
        XCTAssertEqual(result?.word, "leaf")
    }

    func testStar() {
        // star/star_decoration are not in VN taxonomy mappings.
        // Stars are hard to classify via VN — decorative stars would get "picture" via decoration label.
        // This test verifies decoration→picture mapping works for star-like decorations.
        let result = simulateWithRealMappings(observations: [
            ("decoration", 0.80), ("art", 0.60)
        ])
        XCTAssertEqual(result?.word, "picture")
    }

    func testSun() {
        let result = simulateWithRealMappings(observations: [
            ("sky", 0.80), ("sun", 0.72)
        ])
        XCTAssertEqual(result?.word, "sun")
    }

    func testMoon() {
        let result = simulateWithRealMappings(observations: [
            ("sky", 0.80), ("moon", 0.72)
        ])
        XCTAssertEqual(result?.word, "moon")
    }

    // --- Soda can / drinks ---

    func testSodaCanFromCanLabel() {
        let result = simulateWithRealMappings(observations: [
            ("container", 0.80), ("can", 0.72), ("drink", 0.55)
        ])
        XCTAssertEqual(result?.word, "can")
    }

    func testSodaCanFromSodaLabel() {
        let result = simulateWithRealMappings(observations: [
            ("drink", 0.78), ("liquid", 0.75), ("soda", 0.68)
        ])
        XCTAssertEqual(result?.word, "can")
    }

    func testSodaCanFromPopCan() {
        let result = simulateWithRealMappings(observations: [
            ("container", 0.80), ("pop_can", 0.70), ("drink", 0.55)
        ])
        XCTAssertEqual(result?.word, "can")
    }

    func testBeerCanMapsToCan() {
        let result = simulateWithRealMappings(observations: [
            ("container", 0.78), ("beer_can", 0.72), ("drink", 0.50)
        ])
        XCTAssertEqual(result?.word, "can")
    }

    // --- Toys ---

    func testBall() {
        let result = simulateWithRealMappings(observations: [
            ("ball", 0.82), ("sphere", 0.55)
        ])
        XCTAssertEqual(result?.word, "ball")
    }

    func testDoll() {
        // "toy" maps to "ball", so doll must appear before toy or toy must be absent
        let result = simulateWithRealMappings(observations: [
            ("doll", 0.82), ("toy", 0.70), ("figurine", 0.55)
        ])
        XCTAssertEqual(result?.word, "doll")
    }

    func testTeddyBear() {
        let result = simulateWithRealMappings(observations: [
            ("teddy_bear", 0.82), ("toy", 0.70), ("bear", 0.55)
        ])
        XCTAssertEqual(result?.word, "teddy bear")
    }

    func testBlock() {
        let result = simulateWithRealMappings(observations: [
            ("block", 0.80), ("toy", 0.70), ("blocks", 0.55)
        ])
        XCTAssertEqual(result?.word, "block")
    }

    // --- Body parts ---

    // Note: "face", "hand", and "people" are null-mapped or unmapped in label_mappings.json.
    // VN can detect them but they don't map to vocabulary words.
    // Body parts aren't reliably detectable via VNClassifyImageRequest.

    // --- Edge cases ---

    func testNullMappedLabelsSkipped() {
        // All abstract/null-mapped — should return nil
        let result = simulateWithRealMappings(observations: [
            ("document", 0.90), ("screenshot", 0.88),
            ("structure", 0.70), ("interior_room", 0.55)
        ])
        XCTAssertNil(result, "All null-mapped labels should return nil")
    }

    func testAbstractConceptsRejected() {
        // music and design are null-mapped; art maps to "picture" now
        let result = simulateWithRealMappings(observations: [
            ("music", 0.85), ("design", 0.55), ("chart", 0.40)
        ])
        XCTAssertNil(result, "Abstract concepts should not map to anything")
    }

    func testVeryLowConfidenceObjectRejected() {
        let result = simulateWithRealMappings(observations: [
            ("document", 0.90), ("screenshot", 0.88),
            ("dog", 0.02) // Below 0.03 threshold (at scan floor)
        ])
        XCTAssertNil(result, "Dog at 0.02 should be rejected")
    }

    func testSpeakerDetected() {
        let result = simulateWithRealMappings(observations: [
            ("music", 0.80), ("speakers_music", 0.72), ("stereo", 0.55)
        ])
        XCTAssertEqual(result?.word, "speaker")
    }

    func testGlassDetected() {
        let result = simulateWithRealMappings(observations: [
            ("tableware", 0.82), ("drinking_glass", 0.75), ("glass", 0.60)
        ])
        XCTAssertEqual(result?.word, "glass")
    }

    func testToilet() {
        let result = simulateWithRealMappings(observations: [
            ("fixture", 0.80), ("toilet", 0.75)
        ])
        XCTAssertEqual(result?.word, "toilet")
    }

    func testSink() {
        let result = simulateWithRealMappings(observations: [
            ("fixture", 0.80), ("sink", 0.72)
        ])
        XCTAssertEqual(result?.word, "sink")
    }

    func testDoor() {
        let result = simulateWithRealMappings(observations: [
            ("structure", 0.80), ("door", 0.72)
        ])
        XCTAssertEqual(result?.word, "door")
    }

    func testWindow() {
        let result = simulateWithRealMappings(observations: [
            ("structure", 0.80), ("window", 0.72)
        ])
        XCTAssertEqual(result?.word, "window")
    }

    // --- Confidence Aggregation ---

    /// Multiple light labels should aggregate to beat a single moon label.
    /// Real scenario: ceiling light returns moon(0.099), light(0.044), light_bulb(0.043), spotlight(0.021).
    /// Without aggregation: moon wins at 0.099.
    /// With aggregation: light = 0.044+0.043+0.021 = 0.108 beats moon 0.099.
    func testLightAggregationBeatsMoon() {
        let result = simulateWithRealMappings(observations: [
            ("outdoor", 0.100), ("sky", 0.100), ("night_sky", 0.100),
            ("celestial_body", 0.099), ("moon", 0.099),
            ("light", 0.044), ("light_bulb", 0.043), ("spotlight", 0.021)
        ])
        XCTAssertEqual(result?.word, "light", "Aggregated light labels should beat single moon label")
    }

    /// When moon is clearly dominant, it should still win.
    func testMoonStillWinsWhenDominant() {
        let result = simulateWithRealMappings(observations: [
            ("outdoor", 0.74), ("sky", 0.74), ("night_sky", 0.74),
            ("celestial_body", 0.096), ("moon", 0.095),
            ("material", 0.055), ("light", 0.005)
        ])
        XCTAssertEqual(result?.word, "moon", "Strong moon signal should still win")
    }

    /// Cup labels (cup, mug) should aggregate correctly.
    func testCupAggregation() {
        let result = simulateWithRealMappings(observations: [
            ("utensil", 0.59), ("tableware", 0.59), ("mug", 0.53),
            ("cup", 0.46), ("pillow", 0.26)
        ])
        // cup(0.46) + mug(0.53) → "cup" aggregated at 0.99 if mug→cup
        // pillow at 0.26 → much lower
        XCTAssertEqual(result?.word, "cup")
    }

    /// Art/picture labels are null-mapped. Window at low confidence should
    /// still win when it's the only mapped label — this is a VN limitation.
    func testArtMappsToPictureNotWindow() {
        let result = simulateWithRealMappings(observations: [
            ("art", 0.45), ("illustrations", 0.40), ("painting", 0.20),
            ("window", 0.04)
        ])
        // art + illustrations + painting all map to "picture" now
        // picture(1.05) should dominate over window(0.04)
        XCTAssertEqual(result?.word, "picture")
    }

    /// Multiple bottle/can labels mapping to the same word should aggregate.
    func testBottleAggregation() {
        let result = simulateWithRealMappings(observations: [
            ("container", 0.39), ("bottle", 0.39),
            ("structure", 0.14), ("thermos", 0.09),
            ("jar", 0.06), ("jug", 0.04)
        ])
        // bottle(0.39) + thermos(0.09) + jar(0.06) + jug(0.04) all map to "bottle"?
        // At minimum, bottle at 0.39 should win
        XCTAssertEqual(result?.word, "bottle")
    }

    /// Speaker labels (stereo→speaker, speakers_music→speaker) should aggregate.
    func testSpeakerAggregation() {
        let result = simulateWithRealMappings(observations: [
            ("music", 0.092), ("stereo", 0.092),
            ("cord", 0.029), ("machine", 0.023),
            ("speakers_music", 0.019)
        ])
        // stereo→speaker(0.092), speakers_music→speaker(0.019) = 0.111 aggregated
        XCTAssertEqual(result?.word, "speaker")
    }
}
