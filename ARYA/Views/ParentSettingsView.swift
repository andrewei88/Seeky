import SwiftUI

/// Parent settings panel, accessible via long-press on the top-right corner.
/// Shows training capture stats and provides export/clear actions.
struct ParentSettingsView: View {
    let trainingCapture: TrainingCapture
    let correctionStore: CorrectionStore
    @ObservedObject var wordProgressStore: WordProgressStore
    @Binding var environmentOverride: WordEnvironment?
    @Binding var selectedLocation: WordLocation?
    @Binding var selectedCategories: Set<String>
    let onStartQuiz: () -> Void
    let onDismiss: () -> Void

    @State private var captureStats: [(word: String, count: Int)] = []
    @State private var totalCaptures = 0
    @State private var isExporting = false
    @State private var showShareSheet = false
    @State private var exportURL: URL?
    @State private var showClearConfirm = false

    var body: some View {
        ZStack {
            // Dimmed background
            Color.black.opacity(0.7)
                .onTapGesture { onDismiss() }

            VStack(spacing: 0) {
                // Header
                HStack {
                    Text("Parent Settings")
                        .font(.system(size: 20, weight: .semibold, design: .rounded))
                        .foregroundColor(.white)
                    Spacer()
                    Button { onDismiss() } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.white.opacity(0.6))
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 20)
                .padding(.bottom, 12)

                Divider().background(Color.white.opacity(0.2))

                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        // Quiz section
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Quiz Mode")
                                .font(.system(size: 16, weight: .medium, design: .rounded))
                                .foregroundColor(.white)

                            let poolSize = wordProgressStore.quizPoolSize
                            if poolSize >= 3 {
                                Text("\(poolSize) words ready for quizzing")
                                    .font(.system(size: 14, design: .rounded))
                                    .foregroundColor(.white.opacity(0.7))

                                Button { onStartQuiz() } label: {
                                    HStack {
                                        Image(systemName: "magnifyingglass")
                                        Text("New Hunt")
                                    }
                                    .font(.system(size: 16, weight: .medium, design: .rounded))
                                    .foregroundColor(.white)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 12)
                                    .background(Color.green.opacity(0.6))
                                    .cornerRadius(10)
                                }
                            } else {
                                Text("Keep exploring to unlock quiz mode. The app needs to reliably identify at least 3 words in your environment first (\(poolSize) so far).")
                                    .font(.system(size: 14, design: .rounded))
                                    .foregroundColor(.white.opacity(0.5))
                            }
                        }

                        // Location picker
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Location")
                                .font(.system(size: 16, weight: .medium, design: .rounded))
                                .foregroundColor(.white)

                            Text("Where are you? Only objects found at this location will appear.")
                                .font(.system(size: 13, design: .rounded))
                                .foregroundColor(.white.opacity(0.5))

                            LazyVGrid(columns: [
                                GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())
                            ], spacing: 8) {
                                locationButton(label: "All", location: nil, icon: "globe")
                                ForEach(WordLocation.allCases, id: \.self) { loc in
                                    locationButton(label: loc.rawValue, location: loc, icon: loc.icon)
                                }
                            }
                        }

                        // Category picker
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Categories")
                                .font(.system(size: 16, weight: .medium, design: .rounded))
                                .foregroundColor(.white)

                            Text("Focus on specific types. Tap to toggle, or leave all off for a mixed hunt.")
                                .font(.system(size: 13, design: .rounded))
                                .foregroundColor(.white.opacity(0.5))

                            let allCategories = WordProgressStore.quizCategories.keys.sorted()
                            LazyVGrid(columns: [
                                GridItem(.flexible()), GridItem(.flexible())
                            ], spacing: 8) {
                                ForEach(allCategories, id: \.self) { cat in
                                    categoryButton(cat)
                                }
                            }
                        }

                        // Training captures section
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Training Captures")
                                .font(.system(size: 16, weight: .medium, design: .rounded))
                                .foregroundColor(.white)

                            if totalCaptures == 0 {
                                Text("No captures yet. Correct misidentified objects using the pencil button to build training data.")
                                    .font(.system(size: 14, design: .rounded))
                                    .foregroundColor(.white.opacity(0.5))
                            } else {
                                Text("\(totalCaptures) images across \(captureStats.count) words")
                                    .font(.system(size: 14, design: .rounded))
                                    .foregroundColor(.white.opacity(0.7))

                                // Per-word breakdown
                                LazyVGrid(columns: [
                                    GridItem(.flexible(), alignment: .leading),
                                    GridItem(.fixed(40), alignment: .trailing)
                                ], spacing: 4) {
                                    ForEach(captureStats, id: \.word) { stat in
                                        Text(stat.word)
                                            .font(.system(size: 13, design: .rounded))
                                            .foregroundColor(.white.opacity(0.8))
                                        Text("\(stat.count)")
                                            .font(.system(size: 13, weight: .medium, design: .monospaced))
                                            .foregroundColor(.white.opacity(0.6))
                                    }
                                }
                                .padding(.vertical, 8)
                            }
                        }

                        // Learning progress section
                        if wordProgressStore.wordsSeen > 0 {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Learning Progress")
                                    .font(.system(size: 16, weight: .medium, design: .rounded))
                                    .foregroundColor(.white)

                                HStack(spacing: 16) {
                                    progressStat(value: wordProgressStore.wordsSeen, label: "seen")
                                    progressStat(value: wordProgressStore.wordsMastered, label: "mastered")
                                    progressStat(value: wordProgressStore.wordsNeedingPractice, label: "practicing")
                                }

                                let breakdown = wordProgressStore.masteryBreakdown()
                                ForEach(breakdown, id: \.level) { group in
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(masteryLabel(group.level))
                                            .font(.system(size: 12, weight: .medium, design: .rounded))
                                            .foregroundColor(masteryColor(group.level))
                                        Text(group.words.joined(separator: ", "))
                                            .font(.system(size: 12, design: .rounded))
                                            .foregroundColor(.white.opacity(0.6))
                                    }
                                }
                            }
                        }

                        // Corrections section
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Stored Corrections")
                                .font(.system(size: 16, weight: .medium, design: .rounded))
                                .foregroundColor(.white)
                            Text("\(correctionStore.count) correction embeddings")
                                .font(.system(size: 14, design: .rounded))
                                .foregroundColor(.white.opacity(0.7))
                        }

                        // Actions
                        VStack(spacing: 12) {
                            if totalCaptures > 0 {
                                Button {
                                    exportCaptures()
                                } label: {
                                    HStack {
                                        if isExporting {
                                            ProgressView()
                                                .tint(.white)
                                        } else {
                                            Image(systemName: "square.and.arrow.up")
                                        }
                                        Text(isExporting ? "Preparing..." : "Export Captures")
                                    }
                                    .font(.system(size: 16, weight: .medium, design: .rounded))
                                    .foregroundColor(.white)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 12)
                                    .background(Color.blue.opacity(0.6))
                                    .cornerRadius(10)
                                }
                                .disabled(isExporting)
                            }

                            if totalCaptures > 0 || correctionStore.count > 0 || wordProgressStore.wordsSeen > 0 {
                                Button {
                                    showClearConfirm = true
                                } label: {
                                    HStack {
                                        Image(systemName: "trash")
                                        Text("Clear All Data")
                                    }
                                    .font(.system(size: 16, weight: .medium, design: .rounded))
                                    .foregroundColor(.red.opacity(0.8))
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 12)
                                    .background(Color.red.opacity(0.15))
                                    .cornerRadius(10)
                                }
                            }
                        }
                    }
                    .padding(20)
                }
            }
            .frame(maxWidth: 360)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color(white: 0.15))
            )
            .padding(24)
            .alert("Clear All Data?", isPresented: $showClearConfirm) {
                Button("Cancel", role: .cancel) {}
                Button("Clear", role: .destructive) {
                    trainingCapture.clearAll()
                    correctionStore.clearAll()
                    wordProgressStore.clearAll()
                    refreshStats()
                }
            } message: {
                Text("This removes all training captures and stored corrections. Exported zip files are not affected.")
            }
            .sheet(isPresented: $showShareSheet) {
                if let url = exportURL {
                    ShareSheet(url: url)
                }
            }
        }
        .onAppear { refreshStats() }
    }

    private func refreshStats() {
        let result = trainingCapture.stats()
        captureStats = result.perWord
        totalCaptures = result.total
    }

    private func progressStat(value: Int, label: String) -> some View {
        VStack(spacing: 2) {
            Text("\(value)")
                .font(.system(size: 20, weight: .bold, design: .rounded))
                .foregroundColor(.white)
            Text(label)
                .font(.system(size: 11, design: .rounded))
                .foregroundColor(.white.opacity(0.5))
        }
    }

    private func masteryLabel(_ level: Int) -> String {
        switch level {
        case 0: return "New"
        case 1: return "Learning"
        case 2: return "Familiar"
        case 3: return "Strong"
        default: return "Mastered"
        }
    }

    private func masteryColor(_ level: Int) -> Color {
        switch level {
        case 0: return .gray
        case 1: return .orange
        case 2: return .yellow
        case 3: return .blue
        default: return .green
        }
    }

    private func locationButton(label: String, location: WordLocation?, icon: String) -> some View {
        let isSelected = selectedLocation == location
        return Button {
            selectedLocation = location
        } label: {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 16))
                Text(label)
                    .font(.system(size: 11, weight: isSelected ? .semibold : .regular, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .foregroundColor(isSelected ? .white : .white.opacity(0.5))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(isSelected ? Color.blue.opacity(0.5) : Color.white.opacity(0.08))
            .cornerRadius(8)
        }
    }

    private func categoryButton(_ category: String) -> some View {
        let isSelected = selectedCategories.contains(category)
        let count = WordProgressStore.quizCategories[category]?.count ?? 0
        return Button {
            if isSelected {
                selectedCategories.remove(category)
            } else {
                selectedCategories.insert(category)
            }
        } label: {
            HStack {
                Text(category.capitalized)
                    .font(.system(size: 13, weight: isSelected ? .semibold : .regular, design: .rounded))
                Spacer()
                Text("\(count)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.white.opacity(0.4))
            }
            .foregroundColor(isSelected ? .white : .white.opacity(0.5))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(isSelected ? Color.green.opacity(0.4) : Color.white.opacity(0.08))
            .cornerRadius(8)
        }
    }

    private func exportCaptures() {
        isExporting = true
        DispatchQueue.global(qos: .userInitiated).async {
            let url = trainingCapture.createExportArchive()
            DispatchQueue.main.async {
                isExporting = false
                if let url = url {
                    exportURL = url
                    showShareSheet = true
                }
            }
        }
    }
}

/// UIKit share sheet wrapper for SwiftUI.
private struct ShareSheet: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
