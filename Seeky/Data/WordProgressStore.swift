import Foundation

/// Broad locations where a scavenger hunt takes place.
/// Matches the mental model: "We're at the zoo, start a hunt!" not "We're in the kitchen."
enum WordLocation: String, CaseIterable, Codable {
    case home = "Home"
    case body = "Body"
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
        case .body: return "figure.stand"
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

    /// Words the model reliably recognizes at time of seeding.
    /// Accuracy percentages below are from the original model; current R2 model differs.
    /// Words with degraded accuracy are moved to unsafeForQuiz rather than removed from here.
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
        "book", "cherry", "couch", "coconut", "dog", "face", "light", "picture",
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
        "bathtub", "dishwasher", "fridge", "knife", "microwave", "oven",
        "rain", "scissors", "snake", "stairs",
        "toaster", "toilet", "toilet paper",
        // Electronics (TV/monitor separated, both quiz-worthy)
        "laptop", "monitor", "tv",
        // Sports balls (visually distinctive, new in v4)
        "basketball", "soccer ball", "tennis ball",
    ]

    /// Objects excluded from quiz pool because they're unsafe or too unreliable for fair quizzing.
    /// These remain in the classifier for explore-mode identification but are never quiz targets.
    static let unsafeForQuiz: Set<String> = [
        "sun",        // harmful to look at directly
        "book",       // model accuracy too low (72.4% val); per-class threshold at 0.75 blocks most IDs
        "couch",      // model accuracy too low (72.8% val); never reaches 0.40 threshold on device
        "mushroom",   // high false-positive rate on round textured surfaces (stuffed animals, cushions)
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
        "food": ["avocado", "bread", "cake", "cheese", "cookie", "egg", "pizza"],
        "clothing": ["bag", "glasses", "hat", "jacket", "pants", "shirt", "shoe", "sock"],
        "kitchen item": ["bottle", "bowl", "cup", "cupboard", "dishwasher", "fork", "fridge",
                         "glass", "microwave", "pan", "plate", "pot",
                         "spoon", "toaster"],
        "furniture": ["bed", "blanket", "chair", "clock", "door", "fan", "light",
                      "mirror", "picture", "pillow", "shelf", "stairs", "table", "towel",
                      "window"],
        "body part": ["ear", "eye", "face", "foot", "hand", "nose"],
        "vehicle": ["bus", "car", "truck"],
        "toy": ["ball", "basketball", "block", "doll", "soccer ball", "teddy bear", "tennis ball"],
        "bathroom item": ["bathtub", "sink", "soap", "toilet", "toilet paper", "toothbrush"],
        "school supply": ["crayon", "paper", "pen", "pencil"],
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
            "ball", "basketball", "blanket", "block", "box", "cat", "chair", "clock",
            "doll", "dog", "door", "fan", "glasses", "soccer ball",
            "key", "laptop", "light", "mirror", "monitor",
            "phone", "picture", "pillow", "remote", "shelf", "shoe", "speaker",
            "stairs", "table", "teddy bear", "tennis ball", "tv", "umbrella", "window",
            // Bedroom
            "bag", "bed", "hat", "jacket", "pants", "shirt", "sock",
            // Bathroom
            "bathtub", "sink", "soap", "toilet", "toilet paper", "toothbrush", "towel",
            // School supplies (found at home too)
            "crayon", "globe", "keyboard", "paper", "pen", "pencil", "scissors",
        ],
        .body: [
            "ear", "eye", "face", "foot", "hand", "nose",
        ],
        .backyard: [
            "ball", "basketball", "bird", "butterfly", "cat", "cloud", "dog", "door", "fence",
            "flower", "frog", "grass", "leaf", "light", "moon", "rabbit", "rock",
            "soccer ball", "snake", "squirrel", "star", "sunflower", "tennis ball",
            "tree", "turtle", "umbrella", "window",
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

    /// Record that a quiz word was skipped (updates timestamp for cooldown without penalizing).
    func recordQuizSkip(word: String) {
        var entry = progress[word] ?? WordProgress(word: word)
        entry.quizLastDate = Date()
        progress[word] = entry
        save()
        print("[Progress] '\(word)' quiz skipped (cooldown reset)")
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
            if !filtered.isEmpty {
                allEligible = filtered
                print("[Quiz] Location=\(loc.rawValue): \(filtered.count) matching words")
            } else {
                print("[Quiz] Location=\(loc.rawValue): 0 matching, using full pool")
            }
        }

        // Filter by selected categories if specified
        if let cats = categories, !cats.isEmpty {
            let categoryPool = Set(cats.flatMap { Self.quizCategories[$0] ?? [] })
            let filtered = allEligible.intersection(categoryPool)
            if !filtered.isEmpty {
                allEligible = filtered
                print("[Quiz] Categories=\(cats.sorted()): \(filtered.count) matching words")
            } else {
                print("[Quiz] Categories=\(cats.sorted()): 0 matching, expanded to full pool")
            }
        }

        return selectFromPool(eligible: allEligible, count: count)
    }

    /// Shared selection logic: picks words from an eligible set with spaced repetition priority.
    ///
    /// Priority tiers (fill slots in order):
    /// 1. Due for review: words the child has actually answered (correct or wrong) and enough
    ///    time has passed based on mastery level. Skipped-only words are excluded.
    /// 2. Needs practice: words with wrong answers at low mastery.
    /// 3a. Never seen: words that have never appeared in any quiz session. Shuffled for variety.
    /// 3b. Skipped only: words that were skipped but never answered. Oldest-skipped first to
    ///     maximize rotation when the pool is small (e.g., 12 furniture words).
    /// 4. Remaining: anything left, shuffled.
    private func selectFromPool(eligible: Set<String>, count: Int) -> [String] {
        guard !eligible.isEmpty else { return [] }
        var selected: [String] = []
        let now = Date()

        // Tier 1: Words the child has actually answered and are due for review.
        // Only includes words with at least one correct or wrong answer (not just skipped).
        let dueForReview = progress.values
            .filter { eligible.contains($0.word) && $0.quizLastDate != nil }
            .filter { entry in
                // Must have been actually answered, not just skipped
                guard entry.quizCorrect > 0 || entry.quizWrong > 0 else { return false }
                let level = min(entry.masteryLevel, Self.reviewIntervals.count - 1)
                guard now.timeIntervalSince(entry.quizLastDate!) >= Self.reviewIntervals[level] else { return false }
                // Cooldown for mastery-0 words with no correct streak: wait 1 hour.
                if entry.masteryLevel == 0, entry.quizConsecutiveCorrect == 0,
                   entry.quizWrong > 0,
                   now.timeIntervalSince(entry.quizLastDate!) < 3600 {
                    return false
                }
                return true
            }
            .sorted { ($0.quizLastDate ?? .distantPast) < ($1.quizLastDate ?? .distantPast) }
            .map(\.word)
        for word in dueForReview where selected.count < count { selected.append(word) }

        // Tier 2: Words that need practice (have wrong answers), with cooldown for failures.
        let needsPractice = progress.values
            .filter { entry in
                guard eligible.contains(entry.word), entry.quizWrong > 0, !selected.contains(entry.word) else { return false }
                if entry.masteryLevel == 0, entry.quizConsecutiveCorrect == 0,
                   let lastDate = entry.quizLastDate,
                   now.timeIntervalSince(lastDate) < 3600 {
                    return false
                }
                return true
            }
            .sorted { $0.masteryLevel < $1.masteryLevel }
            .map(\.word)
        for word in needsPractice where selected.count < count { selected.append(word) }

        // Tier 3a: Truly fresh — never appeared in any quiz session (no quizLastDate).
        // Shuffled for variety. These get priority over previously-skipped words.
        let neverAnswered = eligible.filter { word in
            guard !selected.contains(word) else { return false }
            guard let entry = progress[word] else { return true }
            return entry.quizCorrect == 0 && entry.quizWrong == 0
        }
        let neverSeen = neverAnswered.filter { progress[$0]?.quizLastDate == nil }.shuffled()
        for word in neverSeen where selected.count < count { selected.append(word) }

        // Tier 3b: Skipped but never answered — sort by oldest quizLastDate first.
        // This rotates through the pool: if you skip [A,B,C] in session 1,
        // session 2 draws from unseen words first, then oldest-skipped.
        let skippedOnly = neverAnswered
            .filter { progress[$0]?.quizLastDate != nil && !selected.contains($0) }
            .sorted { (progress[$0]?.quizLastDate ?? .distantPast) < (progress[$1]?.quizLastDate ?? .distantPast) }
        for word in skippedOnly where selected.count < count { selected.append(word) }

        // Tier 4: Everything else, shuffled.
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
