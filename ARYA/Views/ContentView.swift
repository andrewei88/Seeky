import SwiftUI

struct ContentView: View {
    @StateObject private var appState = AppState()

    var body: some View {
        ZStack {
            // Camera feed — always visible
            CameraPreviewView(cameraManager: appState.cameraManager)
                .ignoresSafeArea()
                .onTapGesture { location in
                    let screenSize = UIScreen.main.bounds.size
                    let normalized = CGPoint(
                        x: location.x / screenSize.width,
                        y: location.y / screenSize.height
                    )
                    if case .learning = appState.mode {
                        appState.dismissLearning()
                    } else {
                        appState.handleTap(at: normalized)
                    }
                }

            // Learning overlay
            if case .learning(let word, let instanceIndex) = appState.mode {
                LearningOverlayView(
                    word: word,
                    instanceIndex: instanceIndex,
                    segmentation: appState.latestSegmentation,
                    wordSpeaker: appState.wordSpeaker
                )
                .transition(.opacity.animation(.easeInOut(duration: 0.3)))
            }

            // First-launch hint
            if !appState.hasCompletedFirstTap && appState.mode == .exploring {
                OnboardingHintView()
            }
        }
        .onAppear {
            appState.cameraManager.configure()
            appState.cameraManager.start()
        }
        .preferredColorScheme(.dark)
        .statusBarHidden()
    }
}
