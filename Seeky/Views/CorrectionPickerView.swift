import SwiftUI

struct CorrectionPickerView: View {
    let words: [String]
    let currentWord: String
    let correctionCount: Int
    let onSelect: (String) -> Void
    let onUndo: () -> Void
    let onCancel: () -> Void

    @State private var searchText = ""

    private var filteredWords: [String] {
        if searchText.isEmpty {
            return words
        }
        return words.filter { $0.localizedCaseInsensitiveContains(searchText) }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Button("Cancel") { onCancel() }
                    .foregroundColor(.white)
                Spacer()
                Text("What is this?")
                    .font(.headline)
                    .foregroundColor(.white)
                Spacer()
                // Balance the header
                Text("Cancel").opacity(0)
            }
            .padding()
            .background(Color.black.opacity(0.8))

            // Search
            TextField("Search words...", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.8))

            // Word list
            ScrollView {
                LazyVStack(spacing: 2) {
                    if correctionCount > 0 && searchText.isEmpty {
                        Button {
                            onUndo()
                        } label: {
                            HStack {
                                Image(systemName: "arrow.uturn.backward")
                                Text("Undo last correction")
                                    .font(.system(size: 18, weight: .medium, design: .rounded))
                                Spacer()
                            }
                            .foregroundColor(.orange)
                            .padding(.horizontal, 20)
                            .padding(.vertical, 14)
                        }

                        Divider().background(Color.gray.opacity(0.3))
                    }

                    ForEach(filteredWords, id: \.self) { word in
                        Button {
                            onSelect(word)
                        } label: {
                            HStack {
                                Text(word)
                                    .font(.system(size: 20, weight: .medium, design: .rounded))
                                    .foregroundColor(.white)
                                Spacer()
                                if word == currentWord {
                                    Text("current")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                }
                            }
                            .padding(.horizontal, 20)
                            .padding(.vertical, 14)
                            .background(word == currentWord ? Color.white.opacity(0.1) : Color.clear)
                        }
                    }
                }
            }
            .background(Color.black.opacity(0.9))
        }
    }
}
