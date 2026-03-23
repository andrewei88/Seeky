import Vision
import CoreImage
import UIKit

struct SegmentationResult {
    let instances: [DetectedInstance]
    let observation: VNInstanceMaskObservation?
    let pixelBuffer: CVPixelBuffer
    let requestHandler: VNImageRequestHandler?
}

final class SegmentationEngine {
    func segment(pixelBuffer: CVPixelBuffer) -> SegmentationResult {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        do {
            try handler.perform([request])
        } catch {
            print("[Segmentation] Error: \(error.localizedDescription)")
            return SegmentationResult(instances: [], observation: nil, pixelBuffer: pixelBuffer, requestHandler: nil)
        }

        guard let observation = request.results?.first else {
            return SegmentationResult(instances: [], observation: nil, pixelBuffer: pixelBuffer, requestHandler: nil)
        }

        let allInstances = observation.allInstances
        var detected: [DetectedInstance] = []

        for index in allInstances {
            if let mask = try? observation.generateScaledMaskForImage(forInstances: IndexSet(integer: index), from: handler) {
                let boundingBox = computeBoundingBox(from: mask)
                guard boundingBox.width > 0.01 && boundingBox.height > 0.01 else { continue }
                detected.append(DetectedInstance(id: index, boundingBox: boundingBox))
            }
        }

        return SegmentationResult(instances: detected, observation: observation, pixelBuffer: pixelBuffer, requestHandler: handler)
    }

    private func computeBoundingBox(from maskBuffer: CVPixelBuffer) -> CGRect {
        let width = CVPixelBufferGetWidth(maskBuffer)
        let height = CVPixelBufferGetHeight(maskBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(maskBuffer)

        CVPixelBufferLockBaseAddress(maskBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(maskBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(maskBuffer) else {
            return .zero
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(maskBuffer)
        var minX = width, minY = height, maxX = 0, maxY = 0

        // Mask can be UInt8 or Float32 depending on device/OS version.
        // Float32 format code: kCVPixelFormatType_OneComponent32Float (0x4c303066)
        let isFloat32 = pixelFormat == kCVPixelFormatType_OneComponent32Float

        for y in 0..<height {
            for x in 0..<width {
                let isForeground: Bool
                if isFloat32 {
                    let ptr = (baseAddress + y * bytesPerRow).assumingMemoryBound(to: Float.self)
                    isForeground = ptr[x] > 0.5
                } else {
                    let ptr = (baseAddress + y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
                    isForeground = ptr[x] > 128
                }
                if isForeground {
                    minX = min(minX, x)
                    minY = min(minY, y)
                    maxX = max(maxX, x)
                    maxY = max(maxY, y)
                }
            }
        }

        guard maxX > minX && maxY > minY else { return .zero }

        return CGRect(
            x: CGFloat(minX) / CGFloat(width),
            y: CGFloat(minY) / CGFloat(height),
            width: CGFloat(maxX - minX) / CGFloat(width),
            height: CGFloat(maxY - minY) / CGFloat(height)
        )
    }
}
