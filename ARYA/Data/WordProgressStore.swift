import Foundation

/// Broad locations where a scavenger hunt takes place.
/// Matches the mental model: "We're at the zoo, start a hunt!" not "We're in the kitchen."
enum WordLocation: String, CaseIterable, Codable {
    case home = "Home"
    case backyard = "Backyard"
    case neighborhood = "Neighborhood"
    case zoo = "Zoo"
    case aquarium = "Aquarium"
    case farm = "Farm"
    case beach = "Beach"
    case forest = "Forest"

    var icon: String {
        switch self {
        case .home: return "house.fill"
        case .backyard: return "leaf.fill"
        case .neighborhood: return "car.fill"
        case .zoo: return "pawprint.fill"
        case .aquarium: return "fish.fill"
        case .farm: return "hare.fill"
        case .beach: return "beach.umbrella.fill"
        case .forest: return "tree.fill"
        }
    }
}

/// Tracks per-word learning progress with separate explore and quiz tracking.
///
/// Explore tracking: records when the model identifies a word (with or without correction).
/// This feeds quiz pool eligibility: 2+ successful IDs with zero corrections.
///
/// Quiz tracking: records correct/wrong answers from quiz mode.
/// Mastery levels (0-4) are based on consecutive correct quiz answers.
/// Spaced repetition intervals increase with mastery.
struct WordProgress: Codable, Equatable {
    let word: String

    // Explore-mode tracking (feeds quiz pool eligibility)
    var exploreIdentified: Int = 0
    var exploreCorrected: Int = 0

    // Quiz-mode tracking (feeds mastery / spaced repetition)
    var quizCorrect: Int = 0
    var quizWrong: Int = 0
    var quizConsecutiveCorrect: Int = 0
    var quizLastDate: Date?
    var masteryLevel: Int = 0

    init(word: String) {
        self.word = word
    }

    /// Quiz-eligible if the model can reliably identify this word in the user's environment.
    var isQuizEligible: Bool {
        exploreIdentified >= 2 && exploreCorrected == 0
    }
}

@MainActor
final class WordProgressStore: ObservableObject {
    @Published private(set) var progress: [String: WordProgress] = [:]
    private let fileURL: URL

    /// Minimum explore-mode identifications before a word enters the quiz pool.
    static let quizEligibilityThreshold = 2

    /// Review intervals per mastery level (in seconds).
    /// Level 0: always due, Level 1: 1h, Level 2: 8h, Level 3: 24h, Level 4: 72h
    static let reviewIntervals: [TimeInterval] = [0, 3600, 28800, 86400, 259200]

    /// Words the model reliably recognizes (90%+ val accuracy).
    /// Physical objects common in households, backyards, and neighborhoods.
    /// Excludes: laptop/monitor (confusion pair), sun (harmful to look at)
    static let seededHighConfidenceWords: Set<String> = [
        // 100% val accuracy
        "block", "bus", "butterfly", "can", "cat", "chicken", "clock", "cloud",
        "deer", "duck", "ear", "egg", "fence", "frog", "glasses", "grass",
        "key", "leaf", "moon", "nose", "pants", "phone",
        "soap", "sock", "star", "tiger", "turtle", "umbrella", "zebra",
        // 99%+ val accuracy
        "car", "cup", "eye", "fan", "giraffe", "lion", "mirror", "monkey",
        "pen", "pencil", "pig", "pineapple", "pizza", "pot",
        "sheep", "strawberry", "towel", "truck",
        // 98%+ val accuracy
        "ball", "bread", "cookie", "crayon", "fish", "fork", "pan",
        "rock", "sink", "squirrel", "tree",
        // 97%+ val accuracy
        "bag", "banana", "bottle", "elephant", "grape", "hat",
        "keyboard", "mango", "orange", "pillow", "plate", "rabbit",
        "remote", "shoe", "speaker", "spoon", "toothbrush",
        // 96%+ val accuracy
        "bear", "blanket", "bowl", "cheese", "cow", "horse", "penguin",
        "shelf", "shirt", "watermelon", "window",
        // 95%+ val accuracy
        "book", "cherry", "couch", "coconut", "dog", "light", "picture",
        // 94%+ val accuracy
        "bench", "cake", "hand",
        // 93%+ val accuracy
        "foot", "octopus",
        // 92%+ val accuracy
        "crab", "fox",
        // 91%+ val accuracy
        "bird", "box", "flower",
        // 90%+ val accuracy
        "apple", "bed", "teddy bear",
        // Household / environment objects (included for quiz)
        "bathtub", "dishwasher", "knife", "microwave", "oven",
        "rain", "scissors", "snake", "stairs",
        "toaster", "toilet", "toilet paper",
    ]

    /// Objects excluded from quiz pool because they're unsafe or too unreliable for fair quizzing.
    /// These remain in the classifier for explore-mode identification but are never quiz targets.
    static let unsafeForQuiz: Set<String> = [
        "sun",      // harmful to look at directly
        "knife",    // unsafe for toddler to seek out
        "scissors", // unsafe for toddler to seek out
        "oven",     // 80.4% accuracy, confused with microwave/cupboard/dishwasher
    ]

    /// Quiz-friendly category groups for category challenges ("Find an animal").
    /// Each category must have 3+ quizzable words to be viable.
    nonisolated static let quizCategories: [String: Set<String>] = [
        "animal": ["bear", "bird", "butterfly", "camel", "cat", "chicken", "cow", "crab",
                   "deer", "dog", "dolphin", "duck", "elephant", "fish", "fox", "frog",
                   "giraffe", "goat", "hippo", "horse", "jellyfish", "lion", "monkey",
                   "octopus", "owl", "panda", "parrot", "penguin", "pig", "rabbit",
                   "seal", "shark", "sheep", "snake", "squirrel", "tiger", "turtle",
                   "whale", "zebra"],
        "fruit": ["apple", "banana", "cherry", "coconut", "grape", "lemon", "mango",
                  "orange", "peach", "pear", "pineapple", "strawberry", "watermelon"],
        "food": ["avocado", "bread", "cake", "cheese", "cookie", "egg", "mushroom", "pizza"],
        "clothing": ["bag", "glasses", "hat", "jacket", "pants", "shirt", "shoe", "sock"],
        "kitchen item": ["bottle", "bowl", "cup", "cupboard", "dishwasher", "fork", "fridge",
                         "glass", "knife", "microwave", "oven", "pan", "plate", "pot",
                         "spoon", "toaster"],
        "furniture": ["bed", "blanket", "chair", "clock", "couch", "door", "fan", "light",
                      "mirror", "picture", "pillow", "shelf", "stairs", "table", "towel",
                      "window"],
        "body part": ["ear", "eye", "face", "foot", "hand", "nose"],
        "vehicle": ["bus", "car", "truck"],
        "toy": ["ball", "block", "doll", "teddy bear"],
        "bathroom item": ["bathtub", "sink", "soap", "toilet", "toilet paper", "toothbrush"],
        "school supply": ["book", "crayon", "paper", "pen", "pencil", "scissors"],
        "electronics": ["keyboard", "laptop", "phone", "remote", "speaker", "tv"],
    ]

    /// Reverse lookup: word -> category name. Built lazily from quizCategories.
    static let wordToCategory: [String: String] = {
        var map: [String: String] = [:]
        for (category, words) in quizCategories {
            for word in words {
                map[word] = category
            }
        }
        return map
    }()

    /// Words available at each location. A word can appear in multiple locations.
    /// "Home" merges all indoor rooms (kitchen, living room, bedroom, bathroom).
    nonisolated static let locationWords: [WordLocation: Set<String>] = [
        .home: [
            // Kitchen
            "apple", "avocado", "banana", "bottle", "bowl", "bread", "cake", "can",
            "cheese", "cherry", "coconut", "cookie", "cup", "cupboard", "dishwasher",
            "egg", "fork", "fridge", "glass", "grape", "knife", "lemon", "mango",
            "microwave", "mushroom", "orange", "oven", "pan", "peach", "pear",
            "pineapple", "pizza", "plate", "pot", "spoon", "strawberry", "toaster",
            "watermelon",
            // Living room
            "ball", "blanket", "block", "book", "box", "cat", "chair", "clock",
            "couch", "doll", "dog", "door", "ear", "eye", "fan", "foot", "glasses",
            "hand", "key", "laptop", "light", "mirror", "nose",
            "phone", "picture", "pillow", "remote", "shelf", "shoe", "speaker",
            "stairs", "table", "teddy bear", "tv", "umbrella", "window",
            // Bedroom
            "bag", "bed", "hat", "jacket", "pants", "shirt", "sock",
            // Bathroom
            "bathtub", "sink", "soap", "toilet", "toilet paper", "toothbrush", "towel",
            // School supplies (found at home too)
            "crayon", "globe", "keyboard", "paper", "pen", "pencil", "scissors",
        ],
        .backyard: [
            "ball", "bird", "butterfly", "cat", "cloud", "dog", "door", "fence",
            "flower", "frog", "grass", "leaf", "light", "moon", "rabbit", "rock",
            "snake", "squirrel", "star", "sunflower", "tree", "turtle", "umbrella",
            "window",
        ],
        .neighborhood: [
            "bench", "bird", "bus", "car", "cat", "cloud", "dog", "door", "duck",
            "fence", "flower", "grass", "key", "leaf", "light", "moon", "rain",
            "rock", "squirrel", "star", "sun", "tree", "truck", "umbrella", "window",
        ],
        .zoo: [
            "bear", "bird", "butterfly", "camel", "chicken", "cow", "crab", "deer",
            "duck", "elephant", "fish", "fox", "frog", "giraffe", "goat", "hippo",
            "horse", "lion", "monkey", "owl", "panda", "parrot", "penguin", "pig",
            "rabbit", "seal", "shark", "sheep", "snake", "squirrel", "tiger",
            "turtle", "whale", "zebra",
        ],
        .aquarium: [
            "crab", "dolphin", "fish", "frog", "jellyfish", "octopus", "penguin",
            "seal", "shark", "turtle", "whale",
        ],
        .farm: [
            "bird", "cat", "chicken", "cow", "dog", "duck", "egg", "fence", "flower",
            "goat", "grass", "horse", "leaf", "pig", "rabbit", "sheep", "tree",
        ],
        .beach: [
            "ball", "bird", "cloud", "coconut", "crab", "dolphin", "fish", "jellyfish",
            "moon", "octopus", "rock", "seal", "shark", "shell", "star", "sun",
            "sunflower", "tree", "turtle", "umbrella", "whale",
        ],
        .forest: [
            "bear", "bird", "butterfly", "cloud", "deer", "flower", "fox", "frog",
            "grass", "leaf", "moon", "mushroom", "owl", "rabbit", "rain", "rock",
            "snake", "squirrel", "star", "sun", "tree", "turtle",
        ],
    ]

    /// Reverse lookup: word -> set of locations where it can be found.
    nonisolated static let wordLocations: [String: Set<WordLocation>] = {
        var map: [String: Set<WordLocation>] = [:]
        for (location, words) in locationWords {
            for word in words {
                map[word, default: []].insert(location)
            }
        }
        return map
    }()

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        fileURL = docs.appendingPathComponent("word_progress.json")
        load()
    }

    init(fileURL: URL) {
        self.fileURL = fileURL
        load()
    }

    // MARK: - Explore Mode

    /// Record a successful identification in explore mode (model returned a word, no correction).
    func recordExploreIdentification(word: String) {
        var entry = progress[word] ?? WordProgress(word: word)
        entry.exploreIdentified += 1
        progress[word] = entry
        save()
        print("[Progress] '\(word)' explored (identified: \(entry.exploreIdentified), corrected: \(entry.exploreCorrected))")
    }

    /// Record a correction in explore mode (parent fixed the identification).
    func recordExploreCorrection(word: String) {
        var entry = progress[word] ?? WordProgress(word: word)
        entry.exploreCorrected += 1
        progress[word] = entry
        save()
        print("[Progress] '\(word)' explore-corrected (identified: \(entry.exploreIdentified), corrected: \(entry.exploreCorrected))")
    }

    // MARK: - Quiz Mode

    /// Record a correct quiz answer.
    func recordQuizCorrect(word: String) {
        var entry = progress[word] ?? WordProgress(word: word)
        entry.quizCorrect += 1
        entry.quizConsecutiveCorrect += 1
        entry.quizLastDate = Date()
        entry.masteryLevel = Self.masteryLevel(for: entry.quizConsecutiveCorrect)
        progress[word] = entry
        save()
        print("[Progress] '\(word)' quiz correct (streak: \(entry.quizConsecutiveCorrect), mastery: \(entry.masteryLevel))")
    }

    /// Record a wrong quiz answer.
    func recordQuizWrong(word: String) {
        var entry = progress[word] ?? WordProgress(word: word)
        entry.quizWrong += 1
        entry.quizConsecutiveCorrect = 0
        entry.quizLastDate = Date()
        entry.masteryLevel = max(0, entry.masteryLevel - 1)
        progress[word] = entry
        save()
        print("[Progress] '\(word)' quiz wrong (mastery: \(entry.masteryLevel))")
    }

    // MARK: - Quiz Pool

    /// Words eligible for quiz mode: seeded high-confidence words minus unsafe ones.
    /// Explore-proven words are disabled until model accuracy improves (laptop/monitor confusion etc).
    func quizEligibleWords() -> [String] {
        return Self.seededHighConfidenceWords.subtracting(Self.unsafeForQuiz).sorted()
    }

    /// Select words for a quiz session, filtered by location and/or categories.
    func selectQuizWords(count: Int = 5, location: WordLocation? = nil, categories: Set<String>? = nil) -> [String] {
        var allEligible = Set(quizEligibleWords())

        // Filter by location if specified
        if let loc = location {
            let locationPool = Self.locationWords[loc] ?? []
            let filtered = allEligible.intersection(locationPool)
            if filtered.count >= count {
                allEligible = filtered
                print("[Quiz] Location=\(loc.rawValue): \(filtered.count) matching words")
            } else {
                print("[Quiz] Location=\(loc.rawValue): only \(filtered.count) matching, using full pool")
            }
        }

        // Filter by selected categories if specified
        if let cats = categories, !cats.isEmpty {
            let categoryPool = Set(cats.flatMap { Self.quizCategories[$0] ?? [] })
            let filtered = allEligible.intersection(categoryPool)
            if filtered.count >= count {
                allEligible = filtered
                print("[Quiz] Categories=\(cats.sorted()): \(filtered.count) matching words")
            } else {
                print("[Quiz] Categories=\(cats.sorted()): only \(filtered.count) matching, expanded to full pool")
            }
        }

        return selectFromPool(eligible: allEligible, count: count)
    }

    /// Shared selection logic: picks words from an eligible set with spaced repetition priority.
    private func selectFromPool(eligible: Set<String>, count: Int) -> [String] {
        guard !eligible.isEmpty else { return [] }
        var selected: [String] = []
        let now = Date()

        let dueForReview = progress.values
            .filter { eligible.contains($0.word) && $0.quizLastDate != nil }
            .filter { entry in
                let level = min(entry.masteryLevel, Self.reviewIntervals.count - 1)
                return now.timeIntervalSince(entry.quizLastDate!) >= Self.reviewIntervals[level]
            }
            .sorted { ($0.quizLastDate ?? .distantPast) < ($1.quizLastDate ?? .distantPast) }
            .map(\.word)
        for word in dueForReview where selected.count < count { selected.append(word) }

        let needsPractice = progress.values
            .filter { eligible.contains($0.word) && $0.quizWrong > 0 && !selected.contains($0.word) }
            .sorted { $0.masteryLevel < $1.masteryLevel }
            .map(\.word)
        for word in needsPractice where selected.count < count { selected.append(word) }

        let neverQuizzed = eligible
            .filter { word in
                guard let entry = progress[word] else { return !selected.contains(word) }
                return entry.quizCorrect == 0 && entry.quizWrong == 0 && !selected.contains(word)
            }
            .shuffled()
        for word in neverQuizzed where selected.count < count { selected.append(word) }

        let remaining = eligible.filter { !selected.contains($0) }.shuffled()
        for word in remaining where selected.count < count { selected.append(word) }

        return Array(selected.prefix(count)).shuffled()
    }

    // MARK: - Stats

    /// Words the child has seen in explore mode.
    var wordsSeen: Int { progress.values.filter { $0.exploreIdentified > 0 }.count }

    /// Words eligible for quiz (seeded + explore-proven).
    var quizPoolSize: Int { quizEligibleWords().count }

    /// Words at mastery level 4 (from quiz results).
    var wordsMastered: Int { progress.values.filter { $0.masteryLevel >= 4 }.count }

    /// Words quizzed but below mastery level 2.
    var wordsNeedingPractice: Int {
        progress.values.filter { ($0.quizCorrect > 0 || $0.quizWrong > 0) && $0.masteryLevel < 2 }.count
    }

    /// Summary grouped by mastery level (only words that have been quizzed).
    func masteryBreakdown() -> [(level: Int, words: [String])] {
        var groups: [Int: [String]] = [:]
        for entry in progress.values where entry.quizCorrect > 0 || entry.quizWrong > 0 {
            groups[entry.masteryLevel, default: []].append(entry.word)
        }
        return groups.sorted { $0.key < $1.key }.map { (level: $0.key, words: $0.value.sorted()) }
    }

    func clearAll() {
        progress.removeAll()
        save()
        print("[Progress] Cleared all word progress")
    }

    static func masteryLevel(for consecutiveCorrect: Int) -> Int {
        switch consecutiveCorrect {
        case 0: return 0
        case 1: return 1
        case 2...3: return 2
        case 4...7: return 3
        default: return 4
        }
    }

    private func save() {
        do {
            let entries = Array(progress.values)
            let data = try JSONEncoder().encode(entries)
            try data.write(to: fileURL)
        } catch {
            print("[Progress] Failed to save: \(error)")
        }
    }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let entries = try? JSONDecoder().decode([WordProgress].self, from: data) else {
            return
        }
        progress = Dictionary(uniqueKeysWithValues: entries.map { ($0.word, $0) })
        print("[Progress] Loaded progress for \(progress.count) words (\(wordsMastered) mastered)")
    }
}
