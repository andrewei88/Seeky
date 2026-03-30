import Foundation

struct PhonemeTimingEntry: Codable {
    let phoneme: String
    let letters: [Int]
    let start: Double
    let end: Double
}

struct TimingData: Codable {
    let word: String
    let phonemes: [PhonemeTimingEntry]

    /// Returns letter indices currently being spoken at the given time.
    func activeLetterIndices(at time: Double) -> [Int] {
        for entry in phonemes {
            if time >= entry.start && time < entry.end {
                return entry.letters
            }
        }
        return []
    }

    /// Returns letter indices that have already been fully spoken at the given time.
    func spokenLetterIndices(at time: Double) -> [Int] {
        var spoken: [Int] = []
        for entry in phonemes {
            if entry.end <= time {
                spoken.append(contentsOf: entry.letters)
            }
        }
        return spoken
    }

    /// Total duration of the word audio based on timing data.
    var duration: Double {
        phonemes.last?.end ?? 0
    }

    static func load(word: String, from bundle: Bundle = .main) -> TimingData? {
        guard let url = bundle.url(forResource: "timing", withExtension: "json", subdirectory: "Vocabulary/\(word)"),
              let data = try? Data(contentsOf: url),
              let timing = try? JSONDecoder().decode(TimingData.self, from: data) else {
            return nil
        }
        return timing
    }
}
