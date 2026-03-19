import Foundation

struct VocabularyEntry: Codable {
    let word: String
    let category: String
}

final class VocabularyStore {
    let entries: [VocabularyEntry]
    let words: Set<String>

    init(entries: [VocabularyEntry]) {
        self.entries = entries
        self.words = Set(entries.map(\.word))
    }

    static func load(from bundle: Bundle = .main) -> VocabularyStore {
        guard let url = bundle.url(forResource: "vocabulary", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let entries = try? JSONDecoder().decode([VocabularyEntry].self, from: data) else {
            fatalError("Failed to load vocabulary.json")
        }
        return VocabularyStore(entries: entries)
    }

    func contains(_ word: String) -> Bool {
        words.contains(word)
    }
}
