import AVFoundation
import Combine

final class WordSpeaker: NSObject, ObservableObject {
    @Published private(set) var currentTime: Double = 0
    @Published private(set) var isPlaying: Bool = false
    @Published private(set) var totalDuration: Double = 0

    /// Playback rate (0.5 = half speed, 1.0 = normal). Applied on top of ElevenLabs 0.7x generation speed.
    var playbackRate: Float = 0.85

    private var audioPlayer: AVAudioPlayer?
    private var displayLink: CADisplayLink?
    private var onComplete: (() -> Void)?

    func speak(word: String, from bundle: Bundle = .main, onComplete: @escaping () -> Void) {
        stop()
        self.onComplete = onComplete

        // Use .playback so audio plays even when ringer is silent.
        // .mixWithOthers prevents interrupting camera capture session (video-only, no mic).
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default, options: [.mixWithOthers])
            try AVAudioSession.sharedInstance().setActive(true)
            print("[WordSpeaker] Audio session configured: category=\(AVAudioSession.sharedInstance().category.rawValue)")
        } catch {
            print("[WordSpeaker] Audio session setup FAILED: \(error)")
        }

        // Load pre-recorded ElevenLabs audio from bundle
        guard let url = bundle.url(forResource: "audio", withExtension: "m4a", subdirectory: "Vocabulary/\(word)") else {
            print("[WordSpeaker] No audio file found for '\(word)', skipping")
            onComplete()
            return
        }

        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = self
            player.enableRate = true
            player.rate = playbackRate
            player.prepareToPlay()
            self.audioPlayer = player
            self.totalDuration = player.duration / Double(playbackRate)
            self.isPlaying = true
            startDisplayLink()
            let started = player.play()
            print("[WordSpeaker] Playing '\(word)': duration=\(String(format: "%.2f", player.duration))s, started=\(started), volume=\(player.volume)")
        } catch {
            print("[WordSpeaker] Failed to create player for '\(word)': \(error)")
            onComplete()
        }
    }

    func stop() {
        audioPlayer?.stop()
        audioPlayer = nil
        stopDisplayLink()
        currentTime = 0
        isPlaying = false
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
}

extension WordSpeaker: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        currentTime = totalDuration
        isPlaying = false
        stopDisplayLink()

        // Brief delay to show all-glow state before dismissing
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.onComplete?()
        }
    }
}
