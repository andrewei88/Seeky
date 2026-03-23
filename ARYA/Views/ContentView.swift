import SwiftUI

struct ContentView: View {
    @StateObject private var appState = AppState()

    var body: some View {
        ZStack {
            // Camera feed — always live
            CameraPreviewView(cameraManager: appState.cameraManager)
                .onTapGesture { location in
                    if appState.showingCorrectionPicker {
                        appState.showingCorrectionPicker = false
                        if case .classifying = appState.mode {
                            appState.mode = .exploring
                        }
                        return
                    }
                    if case .learning = appState.mode {
                        appState.dismissLearning()
                    } else {
                        // Use preview layer to properly convert screen→image coords,
                        // accounting for resizeAspectFill cropping
                        let imagePoint = appState.cameraManager.imagePoint(fromScreenPoint: location)
                        appState.handleTap(imagePoint: imagePoint, screenPoint: location)
                    }
                }

            // Glow at tap point (during learning or unrecognized correction)
            if appState.showGlow {
                TapGlowView(screenPoint: appState.tapScreenPoint)
                    .allowsHitTesting(false)
            }

            // Word display during learning
            if case .learning(let word, _) = appState.mode {
                LearningOverlayView(
                    word: word,
                    wordSpeaker: appState.wordSpeaker
                )
                .allowsHitTesting(false)

                // Correction button for parents (bottom-right)
                if !appState.showingCorrectionPicker {
                    VStack {
                        Spacer()
                        HStack {
                            Spacer()
                            Button {
                                appState.startCorrection()
                            } label: {
                                Image(systemName: "pencil.circle.fill")
                                    .font(.system(size: 36))
                                    .foregroundColor(.white.opacity(0.6))
                            }
                            .padding(.trailing, 24)
                            .padding(.bottom, 60)
                        }
                    }
                }
            }

            // Correction picker overlay
            if appState.showingCorrectionPicker {
                CorrectionPickerView(
                    words: appState.vocabularyStore.sortedWords,
                    currentWord: {
                        if case .learning(let word, _) = appState.mode { return word }
                        return ""
                    }(),
                    onSelect: { word in
                        appState.applyCorrection(word: word)
                    },
                    onCancel: {
                        appState.showingCorrectionPicker = false
                        if case .classifying = appState.mode {
                            appState.mode = .exploring
                        }
                    }
                )
                .transition(.move(edge: .bottom))
            }

            // First-launch hint
            if !appState.hasCompletedFirstTap && appState.mode == .exploring {
                OnboardingHintView()
            }
        }
        .ignoresSafeArea()
        .onAppear {
            appState.cameraManager.configure()
            appState.cameraManager.start()
        }
        .preferredColorScheme(.dark)
        .statusBarHidden()
    }
}
