import XCTest
import AVFoundation
@testable import ARYA

/// Tests that verify bundle resources are accessible the way the app loads them.
/// These catch path/naming mismatches that compilation can't detect.
final class BundleResourceTests: XCTestCase {

    // The test bundle (ARYATests.xctest) is hosted inside the app bundle,
    // so Bundle.main in tests IS the app bundle. This mirrors runtime behavior.

    func testTimingDataLoadsForKnownWords() {
        let words = ["dog", "cat", "laptop", "cup", "bottle", "chair", "ball"]
        for word in words {
            let timing = TimingData.load(word: word)
            XCTAssertNotNil(timing, "TimingData should load for '\(word)'")
            XCTAssertEqual(timing?.word, word, "TimingData.word should match '\(word)'")
            XCTAssertFalse(timing?.phonemes.isEmpty ?? true, "TimingData for '\(word)' should have phonemes")
        }
    }

    func testAudioFileExistsForKnownWords() {
        let words = ["dog", "cat", "laptop", "cup", "bottle", "chair", "ball"]
        for word in words {
            let url = Bundle.main.url(forResource: "audio", withExtension: "m4a", subdirectory: "Vocabulary/\(word)")
            XCTAssertNotNil(url, "Audio file should exist for '\(word)' at Vocabulary/\(word)/audio.m4a")
        }
    }

    func testAudioFileIsPlayable() throws {
        guard let url = Bundle.main.url(forResource: "audio", withExtension: "m4a", subdirectory: "Vocabulary/dog") else {
            XCTFail("Audio file not found for 'dog'")
            return
        }

        let player = try AVAudioPlayer(contentsOf: url)
        XCTAssertGreaterThan(player.duration, 0, "Audio duration should be > 0")
        XCTAssertTrue(player.prepareToPlay(), "Audio should be preparable for playback")
    }

    func testLabelMappingsLoadFromBundle() {
        // This will fatalError if the file is missing, but that's the existing behavior
        let mapper = LabelMapper.load()
        // Verify some known mappings work
        XCTAssertEqual(mapper.childWord(for: "golden retriever"), "dog")
        XCTAssertEqual(mapper.childWord(for: "laptop"), "laptop")
        XCTAssertNil(mapper.childWord(for: "document"), "document should be null-mapped (rejected)")
    }

    func testVocabularyLoadsFromBundle() {
        let store = VocabularyStore.load()
        XCTAssertTrue(store.contains("dog"))
        XCTAssertTrue(store.contains("laptop"))
        XCTAssertTrue(store.contains("cup"))
        XCTAssertFalse(store.contains("nonexistent"))
    }

    func testEveryVocabularyWordHasAudioAndTiming() {
        let store = VocabularyStore.load()
        var missingAudio: [String] = []
        var missingTiming: [String] = []

        for entry in store.entries {
            let audioURL = Bundle.main.url(forResource: "audio", withExtension: "m4a", subdirectory: "Vocabulary/\(entry.word)")
            if audioURL == nil { missingAudio.append(entry.word) }

            let timing = TimingData.load(word: entry.word)
            if timing == nil { missingTiming.append(entry.word) }
        }

        XCTAssertTrue(missingAudio.isEmpty, "Words missing audio: \(missingAudio)")
        XCTAssertTrue(missingTiming.isEmpty, "Words missing timing: \(missingTiming)")
    }
}
