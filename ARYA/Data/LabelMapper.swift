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
            if let str = value as? String {
                mappings[key] = str
            } else {
                mappings[key] = nil as String?
            }
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
