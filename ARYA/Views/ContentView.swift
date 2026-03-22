import SwiftUI

struct ContentView: View {
    @StateObject private var appState = AppState()

    var body: some View {
        ZStack {
            // Camera feed — always live
            CameraPreviewView(cameraManager: appState.cameraManager)
                .ignoresSafeArea()
                .onTapGesture { location in
                    if case .learning = appState.mode {
                        appState.dismissLearning()
                    } else {
                        // Use preview layer to properly convert screen→image coords,
                        // accounting for resizeAspectFill cropping
                        let imagePoint = appState.cameraManager.imagePoint(fromScreenPoint: location)
                        let screenSize = UIScreen.main.bounds.size
                        let normalizedScreen = CGPoint(
                            x: location.x / screenSize.width,
                            y: location.y / screenSize.height
                        )
                        appState.handleTap(imagePoint: imagePoint, screenPoint: normalizedScreen)
                    }
                }

            // Glow + word display during learning
            if case .learning(let word, _) = appState.mode {
                TapGlowView(screenPoint: appState.tapScreenPoint)
                    .allowsHitTesting(false)

                LearningOverlayView(
                    word: word,
                    wordSpeaker: appState.wordSpeaker
                )
                .allowsHitTesting(false)
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
