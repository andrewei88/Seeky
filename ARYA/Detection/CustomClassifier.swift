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

    private let model: MLModel
    private let classes: [String]  // ordered class list matching model output indices

    init?(bundle: Bundle = .main) {
        // Load ARYAClassifier.mlmodelc (compiled from .mlpackage by Xcode)
        guard let modelURL = bundle.url(forResource: "ARYAClassifier", withExtension: "mlmodelc") else {
            print("[CustomClassifier] ARYAClassifier.mlmodelc not found in bundle")
            return nil
        }

        do {
            self.model = try MLModel(contentsOf: modelURL)
            print("[CustomClassifier] Model loaded successfully")
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
        // Resize to 224x224 as required by MobileNetV3
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

        // Find top-2
        var indices = Array(0..<numClasses)
        indices.sort { probs[$0] > probs[$1] }

        let top1Idx = indices[0]
        let top2Idx = indices[1]

        // Log top 5 for debugging
        let top5 = indices.prefix(5).map { "\(classes[$0])(\(String(format: "%.3f", probs[$0])))" }
        print("[CustomClassifier] Top 5: \(top5.joined(separator: ", "))")

        return CustomClassifierResult(
            word: classes[top1Idx],
            confidence: Double(probs[top1Idx]),
            secondConfidence: Double(probs[top2Idx]),
            features: features
        )
    }
}
