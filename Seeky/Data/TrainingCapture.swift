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
    let baseDir: URL

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        baseDir = docs.appendingPathComponent("training_captures")
    }

    init(baseDir: URL) {
        self.baseDir = baseDir
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

    /// Returns per-word capture counts and total.
    func stats() -> (perWord: [(word: String, count: Int)], total: Int) {
        let fm = FileManager.default
        guard let wordDirs = try? fm.contentsOfDirectory(at: baseDir, includingPropertiesForKeys: nil,
                                                          options: .skipsHiddenFiles) else {
            return ([], 0)
        }

        var results: [(word: String, count: Int)] = []
        var total = 0

        for dir in wordDirs where dir.hasDirectoryPath {
            let word = dir.lastPathComponent.replacingOccurrences(of: "_", with: " ")
            let files = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil,
                                                      options: .skipsHiddenFiles)) ?? []
            let jpgCount = files.filter { $0.pathExtension.lowercased() == "jpg" }.count
            if jpgCount > 0 {
                results.append((word: word, count: jpgCount))
                total += jpgCount
            }
        }

        results.sort { $0.word < $1.word }
        return (results, total)
    }

    /// Create a zip archive of all captures. Returns the zip file URL, or nil if no captures exist.
    func createExportArchive() -> URL? {
        let (_, total) = stats()
        guard total > 0 else { return nil }

        let tempDir = FileManager.default.temporaryDirectory
        let zipURL = tempDir.appendingPathComponent("seeky_training_captures.zip")

        // Remove old export if it exists
        try? FileManager.default.removeItem(at: zipURL)

        guard let archive = createZip(sourceDir: baseDir, destURL: zipURL) else {
            return nil
        }
        return archive
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

    /// Create a zip file from a directory using NSFileCoordinator.
    /// iOS provides built-in zip support via NSFileCoordinator with .forUploading intent.
    private func createZip(sourceDir: URL, destURL: URL) -> URL? {
        var error: NSError?
        var resultURL: URL?
        let coordinator = NSFileCoordinator()

        // NSFileCoordinator.coordinate with .forUploading on a directory creates a zip
        coordinator.coordinate(readingItemAt: sourceDir, options: .forUploading, error: &error) { zipURL in
            do {
                try FileManager.default.copyItem(at: zipURL, to: destURL)
                resultURL = destURL
            } catch {
                print("[TrainingCapture] Failed to copy zip: \(error)")
            }
        }

        if let error = error {
            print("[TrainingCapture] Failed to create zip: \(error)")
            return nil
        }

        return resultURL
    }
}
