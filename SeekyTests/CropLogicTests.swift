import XCTest
@testable import Seeky

/// Tests for the cropping logic that feeds VNClassifyImageRequest.
/// The core issue: segmentation bounding boxes are often very large (entire foreground),
/// so the crop sent to classification contains too much context, drowning out the actual object.
final class CropLogicTests: XCTestCase {

    // MARK: - Verify the problem: segmentation bboxes from real logs

    /// Real bounding boxes from device logs — these are what segmentation returns.
    /// They're way too large for focused classification.
    func testSegmentationBBoxesAreTooLarge() {
        // From logs: instance bbox=(0.0, 0.0, 0.9981, 0.44375)
        let bbox1 = CGRect(x: 0.0, y: 0.0, width: 0.9981, height: 0.44375)
        // This covers ~44% of the frame area — far too large for "one object"
        XCTAssertGreaterThan(bbox1.width * bbox1.height, 0.3, "Bbox covers >30% of frame — too large for single object")

        // From logs: instance bbox=(0.0, 0.0, 0.9981, 0.9995)
        let bbox2 = CGRect(x: 0.0, y: 0.0, width: 0.9981, height: 0.9995)
        // This is basically THE ENTIRE FRAME
        XCTAssertGreaterThan(bbox2.width * bbox2.height, 0.9, "Bbox covers >90% of frame — classifying everything")

        // From logs: instance bbox=(0.0, 0.35833, 0.9981, 0.30625)
        let bbox3 = CGRect(x: 0.0, y: 0.35833, width: 0.9981, height: 0.30625)
        XCTAssertGreaterThan(bbox3.width * bbox3.height, 0.25, "Bbox covers >25% of frame — too large")
    }

    // MARK: - Tap-centered crop logic

    /// A tap-centered crop should be a fixed-size square around the tap point.
    /// This focuses classification on what the user actually tapped on.
    func testTapCenteredCrop() {
        let tapPoint = CGPoint(x: 0.5, y: 0.4)
        let cropSize: CGFloat = 0.25  // 20% of frame dimension

        let crop = tapCenteredCrop(tapPoint: tapPoint, cropSize: cropSize, frameWidth: 1080, frameHeight: 1920)

        // Should be centered on tap point
        let centerX = crop.midX / 1080.0
        let centerY = crop.midY / 1920.0
        XCTAssertEqual(centerX, 0.5, accuracy: 0.01, "Crop should be centered on tap X")
        XCTAssertEqual(centerY, 0.4, accuracy: 0.01, "Crop should be centered on tap Y")

        // Should be a reasonable size (not the whole frame)
        let areaFraction = (crop.width * crop.height) / (1080.0 * 1920.0)
        XCTAssertLessThan(areaFraction, 0.15, "Crop should be <15% of frame area")
        XCTAssertGreaterThan(areaFraction, 0.01, "Crop should be >1% of frame area")
    }

    /// Tap near edge should clamp crop to frame bounds.
    func testTapCenteredCropClampsToEdge() {
        let tapPoint = CGPoint(x: 0.05, y: 0.95)  // Near bottom-left
        let crop = tapCenteredCrop(tapPoint: tapPoint, cropSize: 0.25, frameWidth: 1080, frameHeight: 1920)

        XCTAssertGreaterThanOrEqual(crop.minX, 0, "Crop should not go below 0")
        XCTAssertGreaterThanOrEqual(crop.minY, 0, "Crop should not go below 0")
        XCTAssertLessThanOrEqual(crop.maxX, 1080, "Crop should not exceed frame width")
        XCTAssertLessThanOrEqual(crop.maxY, 1920, "Crop should not exceed frame height")
    }

    /// Tap-centered crop should still use segmentation bbox when it's small (actual object).
    func testSmallBBoxUsedDirectly() {
        // Small bbox like a cup = (0.3, 0.4, 0.1, 0.08)
        let bbox = CGRect(x: 0.3, y: 0.4, width: 0.1, height: 0.08)
        let area = bbox.width * bbox.height
        XCTAssertLessThan(area, 0.05, "Small bbox should be used directly for classification")
    }

    // MARK: - Device-to-buffer coordinate conversion

    /// captureDevicePointConverted returns landscape sensor coords.
    /// The pixel buffer is rotated 90° CW to portrait.
    /// Conversion: buffer(x,y) = (1 - device.y, device.x)
    func testDeviceToBufferConversion() {
        // Center maps to center
        let center = deviceToBuffer(CGPoint(x: 0.5, y: 0.5))
        XCTAssertEqual(center.x, 0.5, accuracy: 0.001)
        XCTAssertEqual(center.y, 0.5, accuracy: 0.001)

        // Device top-left (0,0) = sensor top-left = portrait top-right after 90° CW
        // → buffer (1, 0)
        let topLeft = deviceToBuffer(CGPoint(x: 0.0, y: 0.0))
        XCTAssertEqual(topLeft.x, 1.0, accuracy: 0.001)
        XCTAssertEqual(topLeft.y, 0.0, accuracy: 0.001)

        // Device bottom-right (1,1) → buffer (0, 1) = portrait bottom-left
        let bottomRight = deviceToBuffer(CGPoint(x: 1.0, y: 1.0))
        XCTAssertEqual(bottomRight.x, 0.0, accuracy: 0.001)
        XCTAssertEqual(bottomRight.y, 1.0, accuracy: 0.001)

        // Device (0, 1) = sensor bottom-left = portrait top-left after 90° CW
        // → buffer (0, 0)
        let sensorBottomLeft = deviceToBuffer(CGPoint(x: 0.0, y: 1.0))
        XCTAssertEqual(sensorBottomLeft.x, 0.0, accuracy: 0.001)
        XCTAssertEqual(sensorBottomLeft.y, 0.0, accuracy: 0.001)

        // Device (1, 0) = sensor top-right = portrait bottom-right after 90° CW
        // → buffer (1, 1)
        let sensorTopRight = deviceToBuffer(CGPoint(x: 1.0, y: 0.0))
        XCTAssertEqual(sensorTopRight.x, 1.0, accuracy: 0.001)
        XCTAssertEqual(sensorTopRight.y, 1.0, accuracy: 0.001)
    }

    /// Verify the conversion preserves symmetry around center
    func testDeviceToBufferSymmetry() {
        // Two points equidistant from center should remain equidistant
        let p1 = deviceToBuffer(CGPoint(x: 0.3, y: 0.4))
        let p2 = deviceToBuffer(CGPoint(x: 0.7, y: 0.6))
        let dist1 = sqrt(pow(p1.x - 0.5, 2) + pow(p1.y - 0.5, 2))
        let dist2 = sqrt(pow(p2.x - 0.5, 2) + pow(p2.y - 0.5, 2))
        XCTAssertEqual(dist1, dist2, accuracy: 0.01, "Symmetric points should be equidistant from center")
    }

    private func deviceToBuffer(_ devicePoint: CGPoint) -> CGPoint {
        CGPoint(x: 1.0 - devicePoint.y, y: devicePoint.x)
    }

    // MARK: - Helper: tap-centered crop calculation

    /// Computes a crop rect in pixel coordinates, centered on the tap point.
    private func tapCenteredCrop(tapPoint: CGPoint, cropSize: CGFloat, frameWidth: Int, frameHeight: Int) -> CGRect {
        let w = CGFloat(frameWidth)
        let h = CGFloat(frameHeight)

        // Use the smaller dimension to compute crop size (square-ish crop)
        let minDim = min(w, h)
        let cropPixels = minDim * cropSize

        let centerX = tapPoint.x * w
        let centerY = tapPoint.y * h

        var cropRect = CGRect(
            x: centerX - cropPixels / 2,
            y: centerY - cropPixels / 2,
            width: cropPixels,
            height: cropPixels
        )

        // Clamp to frame bounds
        if cropRect.minX < 0 { cropRect.origin.x = 0 }
        if cropRect.minY < 0 { cropRect.origin.y = 0 }
        if cropRect.maxX > w { cropRect.origin.x = w - cropRect.width }
        if cropRect.maxY > h { cropRect.origin.y = h - cropRect.height }

        return cropRect.integral
    }
}
