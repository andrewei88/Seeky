import Foundation
import CoreML
import CoreImage

struct CustomClassifierResult {
    let word: String
    let confidence: Double      // softmax probability 0-1
    let secondConfidence: Double
    let features: [Float]       // 1024-dim L2-normalized feature vector (for CorrectionStore)
}

final class CustomClassifier {
    private static let probabilitiesOutputName = "probabilities"
    private static let featuresOutputName = "features"

    /// Classes where the model has separate outputs but the app treats them as one word.
    /// Key = model class name, Value = merged word. Classes not in this map keep their original name.
    static let classMerges: [String: String] = [
        "lamp": "light",
    ]

    private let model: MLModel
    private let classes: [String]  // ordered class list matching model output indices

    init?(bundle: Bundle = .main) {
        // Load ARYAClassifier.mlmodelc (compiled from .mlpackage by Xcode)
        guard let modelURL = bundle.url(forResource: "ARYAClassifier", withExtension: "mlmodelc") else {
            print("[CustomClassifier] ARYAClassifier.mlmodelc not found in bundle")
            return nil
        }

        do {
            let startTime = CFAbsoluteTimeGetCurrent()
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            print("[CustomClassifier] Model loaded in \(String(format: "%.2f", elapsed))s")
        } catch {
            print("[CustomClassifier] Failed to load model: \(error)")
            return nil
        }

        // Load class list from model metadata or from bundled JSON
        if let classesJSON = bundle.url(forResource: "arya_classes", withExtension: "json"),
           let data = try? Data(contentsOf: classesJSON),
           let decoded = try? JSONDecoder().decode([String].self, from: data) {
            self.classes = decoded
            print("[CustomClassifier] Loaded \(decoded.count) classes from arya_classes.json")
        } else if let metadata = model.modelDescription.metadata[.init(rawValue: "classes")] as? String,
                  let data = metadata.data(using: .utf8),
                  let decoded = try? JSONDecoder().decode([String].self, from: data) {
            self.classes = decoded
            print("[CustomClassifier] Loaded \(decoded.count) classes from model metadata")
        } else {
            print("[CustomClassifier] No class list found — cannot initialize")
            return nil
        }
    }

    private var hasLoggedDiagnostics = false

    /// Classify a cropped object image. Returns top results + feature vector.
    func classify(imageBuffer: CVPixelBuffer) -> CustomClassifierResult? {
        // Resize to 224x224 as required by the model
        guard let resizedBuffer = resizePixelBuffer(imageBuffer, to: CGSize(width: 224, height: 224)) else {
            print("[CustomClassifier] Failed to resize buffer to 224x224")
            return nil
        }

        // Run inference
        guard let input = try? MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(pixelBuffer: resizedBuffer)]
        ),
              let output = try? model.prediction(from: input) else {
            print("[CustomClassifier] Model prediction failed")
            return nil
        }

        // Extract probabilities
        guard let probsFeature = output.featureValue(for: Self.probabilitiesOutputName),
              let probsArray = probsFeature.multiArrayValue else {
            print("[CustomClassifier] Missing 'probabilities' output")
            return nil
        }

        // Extract feature vector
        guard let featsFeature = output.featureValue(for: Self.featuresOutputName),
              let featsArray = featsFeature.multiArrayValue else {
            print("[CustomClassifier] Missing 'features' output")
            return nil
        }

        // One-time diagnostic log
        if !hasLoggedDiagnostics {
            hasLoggedDiagnostics = true
            let pixelFmt = CVPixelBufferGetPixelFormatType(resizedBuffer)
            let fmtStr = String(format: "0x%08X", pixelFmt)
            print("[CustomClassifier] DIAG: pixelFormat=\(fmtStr), probsType=\(probsArray.dataType.rawValue), featsType=\(featsArray.dataType.rawValue), bufferW=\(CVPixelBufferGetWidth(resizedBuffer)), bufferH=\(CVPixelBufferGetHeight(resizedBuffer))")
        }

        // Read probabilities — use subscript to handle Float16/Float32/Double automatically
        let numClasses = probsArray.count
        guard numClasses == classes.count else {
            print("[CustomClassifier] Class count mismatch: model outputs \(numClasses), expected \(classes.count)")
            return nil
        }

        var probs = [Float](repeating: 0, count: numClasses)
        for i in 0..<numClasses {
            probs[i] = probsArray[i].floatValue
        }

        // Read features — use subscript to handle Float16/Float32/Double automatically
        let featDim = featsArray.count
        var features = [Float](repeating: 0, count: featDim)
        for i in 0..<featDim {
            features[i] = featsArray[i].floatValue
        }

        // Aggregate probabilities for merged classes (e.g., monitor+tv → tv)
        var mergedProbs: [String: Float] = [:]
        for i in 0..<numClasses {
            let rawName = classes[i]
            let mergedName = Self.classMerges[rawName] ?? rawName
            mergedProbs[mergedName, default: 0] += probs[i]
        }

        // Sort merged results by probability
        let sorted = mergedProbs.sorted { $0.value > $1.value }
        let topWord = sorted[0].key
        let topConf = sorted[0].value
        let secondConf = sorted.count > 1 ? sorted[1].value : 0

        // Log top 5 for debugging (show merged probabilities)
        let top5 = sorted.prefix(5).map { "\($0.key)(\(String(format: "%.3f", $0.value)))" }
        print("[CustomClassifier] Top 5: \(top5.joined(separator: ", "))")

        return CustomClassifierResult(
            word: topWord,
            confidence: Double(topConf),
            secondConfidence: Double(secondConf),
            features: features
        )
    }
}
