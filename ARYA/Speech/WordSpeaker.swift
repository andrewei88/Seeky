import AVFoundation
import Combine

final class WordSpeaker: NSObject, ObservableObject {
    @Published private(set) var currentTime: Double = 0
    @Published private(set) var isPlaying: Bool = false

    private var audioPlayer: AVAudioPlayer?
    private var displayLink: CADisplayLink?
    private var onComplete: (() -> Void)?

    func speak(word: String, from bundle: Bundle = .main, onComplete: @escaping () -> Void) {
        stop()
        self.onComplete = onComplete

        guard let url = bundle.url(forResource: "audio", withExtension: "m4a", subdirectory: "Vocabulary/\(word)"),
              let player = try? AVAudioPlayer(contentsOf: url) else {
            onComplete()
            return
        }

        // Configure audio session for playback
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)

        audioPlayer = player
        player.delegate = self
        player.prepareToPlay()
        player.play()
        isPlaying = true

        startDisplayLink()
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
        // Keep final time for "all letters glow" state
        currentTime = player.duration
        isPlaying = false
        stopDisplayLink()

        // Delay before calling completion to show all-glow state
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.onComplete?()
        }
    }
}
