import AVFoundation
import Combine

final class WordSpeaker: NSObject, ObservableObject {
    @Published private(set) var currentTime: Double = 0
    @Published private(set) var isPlaying: Bool = false
    @Published private(set) var totalDuration: Double = 0
    /// True when the word audio (not a prefix like celebration or prompt) is playing.
    @Published private(set) var isPlayingWord: Bool = false

    /// Playback rate (0.5 = half speed, 1.0 = normal). Applied on top of ElevenLabs 0.7x generation speed.
    var playbackRate: Float = 0.85

    private var audioPlayer: AVAudioPlayer?
    private var displayLink: CADisplayLink?
    private var onComplete: (() -> Void)?

    /// Queue of audio URLs to play sequentially (for chained prompts).
    private var audioQueue: [URL] = []

    /// Celebration clip filenames (randomly selected on correct answer).
    private static let celebrationClips = ["yes", "thats_right", "great_job", "you_found_it"]

    /// Play a single word's pre-recorded audio.
    func speak(word: String, from bundle: Bundle = .main, onComplete: @escaping () -> Void) {
        stop()
        self.onComplete = onComplete

        configureAudioSession()

        guard let url = bundle.url(forResource: "audio", withExtension: "m4a", subdirectory: "Vocabulary/\(word)") else {
            print("[WordSpeaker] No audio file found for '\(word)', skipping")
            onComplete()
            return
        }

        isPlayingWord = true
        playURL(url, label: word)
    }

    /// Hunt prompt: "Can you find the..." (pause) then the word.
    func speakHuntPrompt(word: String, from bundle: Bundle = .main, onComplete: @escaping () -> Void) {
        stop()
        self.onComplete = onComplete

        configureAudioSession()

        var urls: [URL] = []

        if let promptURL = bundle.url(forResource: "find_the", withExtension: "m4a", subdirectory: "Vocabulary/_prompts") {
            urls.append(promptURL)
        }
        if let wordURL = bundle.url(forResource: "audio", withExtension: "m4a", subdirectory: "Vocabulary/\(word)") {
            urls.append(wordURL)
        }

        guard !urls.isEmpty else {
            print("[WordSpeaker] No audio for hunt prompt '\(word)'")
            onComplete()
            return
        }

        print("[WordSpeaker] Hunt prompt: 'Can you find the... \(word)'")
        playChain(urls)
    }

    /// Celebration: random celebration clip, then the word (reinforcement).
    func speakCelebration(word: String, from bundle: Bundle = .main, onComplete: @escaping () -> Void) {
        stop()
        self.onComplete = onComplete

        configureAudioSession()

        var urls: [URL] = []

        let clip = Self.celebrationClips.randomElement()!
        if let celebURL = bundle.url(forResource: clip, withExtension: "m4a", subdirectory: "Vocabulary/_prompts") {
            urls.append(celebURL)
        }
        if let wordURL = bundle.url(forResource: "audio", withExtension: "m4a", subdirectory: "Vocabulary/\(word)") {
            urls.append(wordURL)
        }

        guard !urls.isEmpty else {
            print("[WordSpeaker] No audio for celebration '\(word)'")
            onComplete()
            return
        }

        print("[WordSpeaker] Celebration: '\(clip)' + '\(word)'")
        playChain(urls)
    }

    /// Category hunt prompt: plays pre-recorded "Can you find a/an [category]?" audio.
    func speakCategoryPrompt(category: String, from bundle: Bundle = .main, onComplete: @escaping () -> Void) {
        stop()
        self.onComplete = onComplete

        configureAudioSession()

        // Category audio files: find_animal.m4a, find_kitchen_item.m4a, etc.
        let filename = "find_\(category.replacingOccurrences(of: " ", with: "_"))"
        guard let url = bundle.url(forResource: filename, withExtension: "m4a", subdirectory: "Vocabulary/_prompts") else {
            print("[WordSpeaker] No category prompt audio for '\(category)' (expected \(filename).m4a)")
            onComplete()
            return
        }

        isPlaying = true
        print("[WordSpeaker] Category prompt: '\(category)' -> \(filename).m4a")
        playURL(url, label: "category-\(category)")
    }

    func stop() {
        audioPlayer?.stop()
        audioPlayer = nil
        audioQueue.removeAll()
        stopDisplayLink()
        currentTime = 0
        isPlaying = false
        isPlayingWord = false
    }

    // MARK: - Private

    private func playChain(_ urls: [URL]) {
        guard let first = urls.first else { return }
        audioQueue = Array(urls.dropFirst())
        isPlayingWord = audioQueue.isEmpty  // If only one URL, it's the word itself
        playURL(first, label: "chain[\(urls.count)]")
    }

    private func playURL(_ url: URL, label: String) {
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = self
            player.enableRate = true
            player.rate = playbackRate
            player.prepareToPlay()
            self.audioPlayer = player
            self.totalDuration = player.duration
            self.isPlaying = true
            startDisplayLink()
            player.play()
            print("[WordSpeaker] Playing '\(label)': duration=\(String(format: "%.2f", player.duration))s")
        } catch {
            print("[WordSpeaker] Failed to play '\(label)': \(error)")
            isPlaying = false
            onComplete?()
        }
    }

    private func startDisplayLink() {
        stopDisplayLink()
        let link = CADisplayLink(target: self, selector: #selector(displayLinkFired))
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    private func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc private func displayLinkFired() {
        guard let player = audioPlayer else { return }
        currentTime = player.currentTime
    }

    private func configureAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default, options: [.mixWithOthers])
            try AVAudioSession.sharedInstance().setActive(true)
            print("[WordSpeaker] Audio session configured: category=\(AVAudioSession.sharedInstance().category.rawValue)")
        } catch {
            print("[WordSpeaker] Audio session setup FAILED: \(error)")
        }
    }
}

extension WordSpeaker: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        stopDisplayLink()

        // If there are more clips in the chain, play the next one after a brief pause
        if let next = audioQueue.first {
            audioQueue.removeFirst()
            let isLastClip = audioQueue.isEmpty
            let delay: TimeInterval = 0.5
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
                guard let self, self.isPlaying else { return }
                // Set isPlayingWord right before the word audio starts (not during the gap)
                if isLastClip {
                    self.currentTime = 0
                    self.isPlayingWord = true
                }
                self.playURL(next, label: "chain-next")
            }
            return
        }

        // Chain complete
        currentTime = totalDuration
        isPlaying = false

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.onComplete?()
        }
    }
}
