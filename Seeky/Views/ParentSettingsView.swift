import SwiftUI

/// Parent settings panel, accessible via the gear icon.
/// Simplified: category picker, action buttons, progress, collapsible data.
struct ParentSettingsView: View {
    let trainingCapture: TrainingCapture
    let correctionStore: CorrectionStore
    @ObservedObject var wordProgressStore: WordProgressStore
    @Binding var selectedCategory: String?
    let onStartQuiz: () -> Void
    let onExplore: () -> Void
    let onDismiss: () -> Void

    @State private var captureStats: [(word: String, count: Int)] = []
    @State private var totalCaptures = 0
    @State private var isExporting = false
    @State private var showShareSheet = false
    @State private var exportURL: URL?
    @State private var showClearConfirm = false
    @State private var showDataSection = false

    private let allCategories = WordProgressStore.quizCategories.keys.sorted()

    var body: some View {
        ZStack {
            // Dimmed background
            Color.black.opacity(0.7)
                .onTapGesture { onDismiss() }

            VStack(spacing: 0) {
                // Header
                HStack {
                    Text("Settings")
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
                        // Category picker + action buttons
                        VStack(alignment: .leading, spacing: 12) {
                            // Category dropdown
                            HStack {
                                Text("Focus")
                                    .font(.system(size: 16, weight: .medium, design: .rounded))
                                    .foregroundColor(.white)

                                Spacer()

                                Menu {
                                    Button {
                                        selectedCategory = nil
                                    } label: {
                                        HStack {
                                            Text("All categories")
                                            if selectedCategory == nil {
                                                Image(systemName: "checkmark")
                                            }
                                        }
                                    }

                                    Divider()

                                    ForEach(allCategories, id: \.self) { cat in
                                        Button {
                                            selectedCategory = cat
                                        } label: {
                                            HStack {
                                                Text(cat.capitalized)
                                                if selectedCategory == cat {
                                                    Image(systemName: "checkmark")
                                                }
                                            }
                                        }
                                    }
                                } label: {
                                    HStack(spacing: 4) {
                                        Text(selectedCategory?.capitalized ?? "All categories")
                                            .font(.system(size: 15, design: .rounded))
                                        Image(systemName: "chevron.up.chevron.down")
                                            .font(.system(size: 10))
                                    }
                                    .foregroundColor(.white.opacity(0.8))
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(Color.white.opacity(0.12))
                                    .cornerRadius(8)
                                }
                            }

                            // Action buttons
                            let poolSize = wordProgressStore.quizPoolSize
                            if poolSize >= 3 {
                                HStack(spacing: 8) {
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

                                    Button { onExplore() } label: {
                                        HStack {
                                            Image(systemName: "eye")
                                            Text("Explore")
                                        }
                                        .font(.system(size: 16, weight: .medium, design: .rounded))
                                        .foregroundColor(.white)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 12)
                                        .background(Color.blue.opacity(0.4))
                                        .cornerRadius(10)
                                    }
                                }
                            } else {
                                Text("Keep exploring to unlock quiz mode (\(poolSize)/3 words identified).")
                                    .font(.system(size: 14, design: .rounded))
                                    .foregroundColor(.white.opacity(0.5))
                            }
                        }

                        // Learning progress
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

                        // Collapsible data section
                        VStack(alignment: .leading, spacing: 8) {
                            Button {
                                withAnimation(.easeInOut(duration: 0.2)) {
                                    showDataSection.toggle()
                                }
                            } label: {
                                HStack {
                                    Text("Data")
                                        .font(.system(size: 16, weight: .medium, design: .rounded))
                                        .foregroundColor(.white)
                                    Spacer()
                                    let summary = dataSummary
                                    if !summary.isEmpty {
                                        Text(summary)
                                            .font(.system(size: 13, design: .rounded))
                                            .foregroundColor(.white.opacity(0.4))
                                    }
                                    Image(systemName: showDataSection ? "chevron.up" : "chevron.down")
                                        .font(.system(size: 12, weight: .medium))
                                        .foregroundColor(.white.opacity(0.4))
                                }
                            }

                            if showDataSection {
                                VStack(alignment: .leading, spacing: 16) {
                                    if totalCaptures > 0 {
                                        VStack(alignment: .leading, spacing: 4) {
                                            Text("Training Captures")
                                                .font(.system(size: 14, weight: .medium, design: .rounded))
                                                .foregroundColor(.white.opacity(0.8))
                                            Text("\(totalCaptures) images across \(captureStats.count) words")
                                                .font(.system(size: 13, design: .rounded))
                                                .foregroundColor(.white.opacity(0.5))

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
                                            .padding(.vertical, 4)
                                        }
                                    }

                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("Stored Corrections")
                                            .font(.system(size: 14, weight: .medium, design: .rounded))
                                            .foregroundColor(.white.opacity(0.8))
                                        Text("\(correctionStore.count) correction embeddings")
                                            .font(.system(size: 13, design: .rounded))
                                            .foregroundColor(.white.opacity(0.5))
                                    }

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
                                .padding(.top, 4)
                            }
                        }
                    }
                    .padding(20)
                }
            }
            .frame(maxWidth: 360, maxHeight: UIScreen.main.bounds.height - 80)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color(white: 0.15))
            )
            .padding(.horizontal, 24)
            .padding(.vertical, 40)
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

    private var dataSummary: String {
        var parts: [String] = []
        if totalCaptures > 0 { parts.append("\(totalCaptures) captures") }
        if correctionStore.count > 0 { parts.append("\(correctionStore.count) corrections") }
        return parts.joined(separator: ", ")
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
