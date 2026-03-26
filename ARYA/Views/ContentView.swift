import SwiftUI

struct ContentView: View {
    @StateObject private var appState = AppState()

    private var isInQuizMode: Bool {
        switch appState.mode {
        case .quizPrompting, .quizClassifying, .quizResult: return true
        default: return false
        }
    }

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

                    // Quiz mode tap handling
                    if case .quizPrompting = appState.mode {
                        let imagePoint = appState.cameraManager.imagePoint(fromScreenPoint: location)
                        appState.handleQuizTap(imagePoint: imagePoint, screenPoint: location)
                        return
                    }

                    if case .learning = appState.mode {
                        appState.dismissLearning()
                    } else if appState.mode == .exploring {
                        let imagePoint = appState.cameraManager.imagePoint(fromScreenPoint: location)
                        appState.handleTap(imagePoint: imagePoint, screenPoint: location)
                    }
                }

            // Glow at tap point (during learning or unrecognized correction)
            if appState.showGlow {
                TapGlowView(screenPoint: appState.tapScreenPoint)
                    .allowsHitTesting(false)
            }

            // Word display during learning (explore mode)
            if case .learning(let word) = appState.mode {
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

            // Scavenger hunt overlay
            if isInQuizMode, let session = appState.quizSession {
                QuizOverlayView(
                    session: session,
                    wordSpeaker: appState.wordSpeaker,
                    mode: appState.mode,
                    onSkip: { appState.skipQuizWord() },
                    onGoBack: { appState.goBackQuizWord() },
                    onRetry: { appState.retryQuizWord() },
                    onAdvance: { appState.advanceQuiz() },
                    onOverride: { appState.overrideQuizResult() },
                    onNewHunt: { appState.startScavengerHunt() },
                    onExplore: { appState.switchToExplore() },
                    onReplay: { appState.replayQuizWord() }
                )
            }

            // Correction picker overlay
            if appState.showingCorrectionPicker {
                CorrectionPickerView(
                    words: appState.vocabularyStore.sortedWords,
                    currentWord: {
                        if case .learning(let word) = appState.mode { return word }
                        return ""
                    }(),
                    correctionCount: appState.correctionStore.count,
                    onSelect: { word in
                        appState.applyCorrection(word: word)
                    },
                    onUndo: {
                        appState.undoLastCorrection()
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

            // Camera permission denied
            if appState.cameraManager.status == .denied {
                VStack(spacing: 16) {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 48))
                        .foregroundColor(.white.opacity(0.6))
                    Text("Camera access needed")
                        .font(.system(size: 24, weight: .medium, design: .rounded))
                        .foregroundColor(.white)
                    Text("Open Settings to allow camera access")
                        .font(.system(size: 16, design: .rounded))
                        .foregroundColor(.white.opacity(0.6))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)
            }

            // First-launch hint (only in explore mode)
            if !appState.hasCompletedFirstTap && appState.mode == .exploring
                && appState.cameraManager.status != .denied {
                OnboardingHintView()
            }

            // Settings gear (top-right, always visible, subtle)
            VStack {
                HStack {
                    Spacer()
                    Button {
                        appState.showingParentSettings = true
                    } label: {
                        Image(systemName: "gearshape.fill")
                            .font(.system(size: 18))
                            .foregroundColor(.white.opacity(0.25))
                            .frame(width: 44, height: 44)
                    }
                    .padding(.trailing, 8)
                    .padding(.top, 50)
                }
                Spacer()
            }

            // In explore mode: show hunt button (bottom-center)
            if appState.mode == .exploring {
                VStack {
                    Spacer()
                    Button {
                        appState.startScavengerHunt()
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "magnifyingglass")
                                .font(.system(size: 14, weight: .medium))
                            Text("Start Hunt")
                                .font(.system(size: 14, weight: .medium, design: .rounded))
                        }
                        .foregroundColor(.white.opacity(0.5))
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(Color.white.opacity(0.1))
                        .cornerRadius(16)
                    }
                    .padding(.bottom, 50)
                }
            }

            // Parent settings overlay
            if appState.showingParentSettings {
                ParentSettingsView(
                    trainingCapture: appState.trainingCapture,
                    correctionStore: appState.correctionStore,
                    wordProgressStore: appState.wordProgressStore,
                    environmentOverride: $appState.environmentOverride,
                    selectedLocation: $appState.selectedLocation,
                    selectedCategories: $appState.selectedCategories,
                    onStartQuiz: { appState.startScavengerHunt() },
                    onDismiss: { appState.showingParentSettings = false }
                )
            }
        }
        .ignoresSafeArea()
        .onAppear {
            appState.cameraManager.configure()
            appState.cameraManager.start()
            appState.startScavengerHunt()
        }
        .preferredColorScheme(.dark)
        .statusBarHidden()
    }
}
