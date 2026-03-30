import XCTest
import CoreImage
@testable import Seeky

/// Tests that cropPixelBuffer correctly handles CIImage's bottom-left coordinate system.
final class CropPixelBufferTests: XCTestCase {

    /// Create a test pixel buffer filled with a known color pattern.
    /// Top-left quadrant = red, top-right = green, bottom-left = blue, bottom-right = white.
    private func makeTestBuffer(width: Int, height: Int) -> CVPixelBuffer {
        var buffer: CVPixelBuffer!
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA, nil, &buffer)
        precondition(status == kCVReturnSuccess)

        CVPixelBufferLockBaseAddress(buffer, [])
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                let isTop = y < height / 2
                let isLeft = x < width / 2

                // BGRA format
                if isTop && isLeft {
                    // Red (B=0, G=0, R=255, A=255)
                    base[offset] = 0; base[offset+1] = 0; base[offset+2] = 255; base[offset+3] = 255
                } else if isTop && !isLeft {
                    // Green (B=0, G=255, R=0, A=255)
                    base[offset] = 0; base[offset+1] = 255; base[offset+2] = 0; base[offset+3] = 255
                } else if !isTop && isLeft {
                    // Blue (B=255, G=0, R=0, A=255)
                    base[offset] = 255; base[offset+1] = 0; base[offset+2] = 0; base[offset+3] = 255
                } else {
                    // White (B=255, G=255, R=255, A=255)
                    base[offset] = 255; base[offset+1] = 255; base[offset+2] = 255; base[offset+3] = 255
                }
            }
        }

        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }

    /// Read the average color of a pixel buffer (returns BGRA).
    private func averageColor(of buffer: CVPixelBuffer) -> (b: UInt8, g: UInt8, r: UInt8, a: UInt8) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

        var totalB: Int = 0, totalG: Int = 0, totalR: Int = 0, totalA: Int = 0
        let count = w * h

        for y in 0..<h {
            for x in 0..<w {
                let offset = y * bytesPerRow + x * 4
                totalB += Int(base[offset])
                totalG += Int(base[offset+1])
                totalR += Int(base[offset+2])
                totalA += Int(base[offset+3])
            }
        }

        return (UInt8(totalB / count), UInt8(totalG / count), UInt8(totalR / count), UInt8(totalA / count))
    }

    /// The crop function under test — extracted from AppState to make it testable.
    /// This MUST match AppState.cropPixelBuffer exactly.
    private func cropPixelBuffer(_ buffer: CVPixelBuffer, to normalizedRect: CGRect) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)

        // CIImage uses bottom-left origin. normalizedRect uses top-left origin.
        let pxX = normalizedRect.origin.x * CGFloat(width)
        let pxW = normalizedRect.width * CGFloat(width)
        let pxH = normalizedRect.height * CGFloat(height)
        let pxY = CGFloat(height) - normalizedRect.origin.y * CGFloat(height) - pxH

        let cropRect = CGRect(x: pxX, y: pxY, width: pxW, height: pxH).integral

        guard cropRect.width > 0, cropRect.height > 0 else { return nil }

        let cropped = CIImage(cvPixelBuffer: buffer)
            .cropped(to: cropRect)
            .transformed(by: CGAffineTransform(translationX: -cropRect.origin.x, y: -cropRect.origin.y))

        let context = CIContext()
        var croppedBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(cropRect.width), Int(cropRect.height),
                           kCVPixelFormatType_32BGRA, nil, &croppedBuffer)
        guard let output = croppedBuffer else { return nil }
        context.render(cropped, to: output)
        return output
    }

    /// Cropping the top-left quadrant should produce a red buffer.
    func testCropTopLeftQuadrant() {
        let buffer = makeTestBuffer(width: 200, height: 200)
        // Top-left quadrant in top-left origin: (0, 0, 0.5, 0.5)
        let result = cropPixelBuffer(buffer, to: CGRect(x: 0, y: 0, width: 0.5, height: 0.5))
        XCTAssertNotNil(result, "cropPixelBuffer should return a valid buffer")

        guard let result = result else { return }
        let color = averageColor(of: result)
        // Should be red: R=255, G=0, B=0
        XCTAssertGreaterThan(color.r, 200, "Top-left quadrant should be red (R channel)")
        XCTAssertLessThan(color.g, 50, "Top-left quadrant should be red (G channel)")
        XCTAssertLessThan(color.b, 50, "Top-left quadrant should be red (B channel)")
    }

    /// Cropping the top-right quadrant should produce a green buffer.
    func testCropTopRightQuadrant() {
        let buffer = makeTestBuffer(width: 200, height: 200)
        let result = cropPixelBuffer(buffer, to: CGRect(x: 0.5, y: 0, width: 0.5, height: 0.5))
        XCTAssertNotNil(result)

        guard let result = result else { return }
        let color = averageColor(of: result)
        XCTAssertLessThan(color.r, 50, "Top-right should be green (R)")
        XCTAssertGreaterThan(color.g, 200, "Top-right should be green (G)")
        XCTAssertLessThan(color.b, 50, "Top-right should be green (B)")
    }

    /// Cropping the bottom-left quadrant should produce a blue buffer.
    func testCropBottomLeftQuadrant() {
        let buffer = makeTestBuffer(width: 200, height: 200)
        let result = cropPixelBuffer(buffer, to: CGRect(x: 0, y: 0.5, width: 0.5, height: 0.5))
        XCTAssertNotNil(result)

        guard let result = result else { return }
        let color = averageColor(of: result)
        XCTAssertLessThan(color.r, 50, "Bottom-left should be blue (R)")
        XCTAssertLessThan(color.g, 50, "Bottom-left should be blue (G)")
        XCTAssertGreaterThan(color.b, 200, "Bottom-left should be blue (B)")
    }

    /// Cropping the bottom-right quadrant should produce a white buffer.
    func testCropBottomRightQuadrant() {
        let buffer = makeTestBuffer(width: 200, height: 200)
        let result = cropPixelBuffer(buffer, to: CGRect(x: 0.5, y: 0.5, width: 0.5, height: 0.5))
        XCTAssertNotNil(result)

        guard let result = result else { return }
        let color = averageColor(of: result)
        XCTAssertGreaterThan(color.r, 200, "Bottom-right should be white (R)")
        XCTAssertGreaterThan(color.g, 200, "Bottom-right should be white (G)")
        XCTAssertGreaterThan(color.b, 200, "Bottom-right should be white (B)")
    }

    /// Full-frame crop should return the entire image (non-nil, correct dimensions).
    func testCropFullFrame() {
        let buffer = makeTestBuffer(width: 200, height: 200)
        let result = cropPixelBuffer(buffer, to: CGRect(x: 0, y: 0, width: 1.0, height: 1.0))
        XCTAssertNotNil(result)

        guard let result = result else { return }
        XCTAssertEqual(CVPixelBufferGetWidth(result), 200)
        XCTAssertEqual(CVPixelBufferGetHeight(result), 200)
    }

    /// Output buffer should have correct dimensions for a center crop.
    func testCropDimensions() {
        let buffer = makeTestBuffer(width: 400, height: 600)
        let result = cropPixelBuffer(buffer, to: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5))
        XCTAssertNotNil(result)

        guard let result = result else { return }
        XCTAssertEqual(CVPixelBufferGetWidth(result), 200)
        XCTAssertEqual(CVPixelBufferGetHeight(result), 300)
    }

    /// Alpha channel should be preserved (non-zero).
    func testCropPreservesAlpha() {
        let buffer = makeTestBuffer(width: 200, height: 200)
        let result = cropPixelBuffer(buffer, to: CGRect(x: 0, y: 0, width: 0.5, height: 0.5))
        XCTAssertNotNil(result)

        guard let result = result else { return }
        let color = averageColor(of: result)
        XCTAssertEqual(color.a, 255, "Alpha should be preserved at 255")
    }
}
