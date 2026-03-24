import UIKit

/// Saves cropped classification images to the app's documents directory for retraining.
///
/// When a user corrects a misidentified object, the cropped image is saved alongside
/// the corrected label. These images can be exported and used to retrain the custom
/// classifier with real phone-camera data that matches the deployment environment.
///
/// Directory structure:
///   Documents/training_captures/{word}/{timestamp}.jpg
final class TrainingCapture {
    private let baseDir: URL

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        baseDir = docs.appendingPathComponent("training_captures")
    }

    /// Save a cropped image buffer with its corrected label.
    func save(imageBuffer: CVPixelBuffer, word: String) {
        let wordDir = baseDir.appendingPathComponent(word.replacingOccurrences(of: " ", with: "_"))

        do {
            try FileManager.default.createDirectory(at: wordDir, withIntermediateDirectories: true)
        } catch {
            print("[TrainingCapture] Failed to create directory: \(error)")
            return
        }

        guard let jpegData = jpegData(from: imageBuffer) else {
            print("[TrainingCapture] Failed to convert buffer to JPEG")
            return
        }

        let timestamp = Int(Date().timeIntervalSince1970 * 1000)
        let fileURL = wordDir.appendingPathComponent("\(timestamp).jpg")

        do {
            try jpegData.write(to: fileURL)
            print("[TrainingCapture] Saved '\(word)' → \(fileURL.lastPathComponent)")
        } catch {
            print("[TrainingCapture] Failed to write image: \(error)")
        }
    }

    /// Remove all captured training data.
    func clearAll() {
        try? FileManager.default.removeItem(at: baseDir)
        print("[TrainingCapture] Cleared all training captures")
    }

    private func jpegData(from buffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        guard let cgImage = sharedCIContext.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        return UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.90)
    }
}
