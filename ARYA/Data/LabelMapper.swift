import Foundation

final class LabelMapper {
    private let mappings: [String: String?]

    init(mappings: [String: String?]) {
        self.mappings = mappings
    }

    static func load(from bundle: Bundle = .main) -> LabelMapper {
        guard let url = bundle.url(forResource: "label_mappings", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            fatalError("Failed to load label_mappings.json")
        }
        var mappings: [String: String?] = [:]
        for (key, value) in raw {
            let mapped: String? = (value as? String)
            // Store the original key
            mappings[key] = mapped
            // Also store normalized variants so VNClassify identifiers match
            // VNClassify may return "golden_retriever" while JSON has "golden retriever"
            let withSpaces = key.replacingOccurrences(of: "_", with: " ")
            let withUnderscores = key.replacingOccurrences(of: " ", with: "_")
            let lowered = key.lowercased()
            mappings[withSpaces] = mapped
            mappings[withUnderscores] = mapped
            mappings[lowered] = mapped
            mappings[lowered.replacingOccurrences(of: " ", with: "_")] = mapped
        }
        return LabelMapper(mappings: mappings)
    }

    /// Returns the child-friendly word for a VNClassify label, or nil if unmapped/rejected.
    func childWord(for classifierLabel: String) -> String? {
        guard let mapping = mappings[classifierLabel] else {
            return nil // Not in whitelist
        }
        return mapping // nil if explicitly rejected, String if mapped
    }
}
