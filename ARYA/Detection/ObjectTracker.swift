import Vision

/// Lightweight object tracker using Vision's VNTrackObjectRequest.
/// Runs at camera frame rate (~30fps), much faster than segmentation (~3fps).
/// Thread-safe — designed to be called from the camera callback thread.
final class ObjectTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var sequenceHandler: VNSequenceRequestHandler?
    private var lastObservation: VNDetectedObjectObservation?

    /// Start tracking an object at the given bounding box.
    /// boundingBox uses top-left origin, normalized 0-1 coordinates.
    func startTracking(boundingBox: CGRect) {
        // Convert from top-left origin to Vision's bottom-left origin
        let visionBBox = CGRect(
            x: boundingBox.origin.x,
            y: 1.0 - boundingBox.origin.y - boundingBox.height,
            width: boundingBox.width,
            height: boundingBox.height
        )
        lock.lock()
        lastObservation = VNDetectedObjectObservation(boundingBox: visionBBox)
        sequenceHandler = VNSequenceRequestHandler()
        lock.unlock()
    }

    /// Stop tracking.
    func stopTracking() {
        lock.lock()
        lastObservation = nil
        sequenceHandler = nil
        lock.unlock()
    }

    /// Whether tracking is currently active.
    var isTracking: Bool {
        lock.lock()
        defer { lock.unlock() }
        return lastObservation != nil
    }

    /// Track the object in the given pixel buffer.
    /// Returns the updated bounding box in top-left origin coordinates, or nil if tracking lost.
    func track(pixelBuffer: CVPixelBuffer) -> CGRect? {
        lock.lock()
        guard let obs = lastObservation, let handler = sequenceHandler else {
            lock.unlock()
            return nil
        }
        lock.unlock()

        let request = VNTrackObjectRequest(detectedObjectObservation: obs)
        request.trackingLevel = .fast

        do {
            try handler.perform([request], on: pixelBuffer)
        } catch {
            print("[Track] Error: \(error.localizedDescription)")
            return nil
        }

        guard let result = request.results?.first as? VNDetectedObjectObservation else {
            return nil
        }

        // If tracking confidence is too low, the object is likely lost
        guard result.confidence > 0.2 else {
            print("[Track] Low confidence: \(String(format: "%.2f", result.confidence))")
            return nil
        }

        // Update observation for next frame
        lock.lock()
        lastObservation = result
        lock.unlock()

        // Convert back from Vision coordinates (bottom-left) to top-left origin
        let bbox = result.boundingBox
        return CGRect(
            x: bbox.origin.x,
            y: 1.0 - bbox.origin.y - bbox.height,
            width: bbox.width,
            height: bbox.height
        )
    }
}
