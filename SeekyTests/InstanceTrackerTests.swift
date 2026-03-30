import XCTest
import CoreGraphics
@testable import ARYA

final class InstanceTrackerTests: XCTestCase {

    func testNewInstanceNotImmediatelyTappable() {
        var tracker = InstanceTracker(stabilityDuration: 0.5)
        tracker.update(instances: [
            DetectedInstance(id: 1, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2))
        ], timestamp: 0.0)

        let tappable = tracker.tappableInstances(at: 0.0)
        XCTAssertTrue(tappable.isEmpty)
    }

    func testInstanceBecomesTappableAfterStabilityDuration() {
        var tracker = InstanceTracker(stabilityDuration: 0.5)
        let instance = DetectedInstance(id: 1, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2))

        tracker.update(instances: [instance], timestamp: 0.0)
        tracker.update(instances: [instance], timestamp: 0.2)
        tracker.update(instances: [instance], timestamp: 0.5)

        let tappable = tracker.tappableInstances(at: 0.5)
        XCTAssertEqual(tappable.count, 1)
    }

    // Grace period: instances survive a brief disappearance and keep their stability timestamp
    func testInstanceSurvivesBriefDisappearance() {
        var tracker = InstanceTracker(stabilityDuration: 0.3, gracePeriod: 1.5)
        let instance = DetectedInstance(id: 1, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2))

        tracker.update(instances: [instance], timestamp: 0.0)
        tracker.update(instances: [instance], timestamp: 0.2)
        // Instance disappears for one frame
        tracker.update(instances: [], timestamp: 0.3)
        // Instance reappears
        tracker.update(instances: [instance], timestamp: 0.4)

        // Should still be tappable — grace period kept it alive, firstSeen is still 0.0
        let tappable = tracker.tappableInstances(at: 0.4)
        XCTAssertEqual(tappable.count, 1, "Instance should survive brief disappearance via grace period")
    }

    func testInstanceRemovedAfterGracePeriodExpires() {
        var tracker = InstanceTracker(stabilityDuration: 0.3, gracePeriod: 0.5)
        let instance = DetectedInstance(id: 1, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2))

        tracker.update(instances: [instance], timestamp: 0.0)
        // Instance disappears
        tracker.update(instances: [], timestamp: 0.1)
        // Still within grace period (0.5s)
        tracker.update(instances: [], timestamp: 0.4)
        XCTAssertEqual(tracker.trackedInstances.count, 1, "Should still be tracked within grace period")

        // Grace period expired
        tracker.update(instances: [], timestamp: 0.7)
        XCTAssertEqual(tracker.trackedInstances.count, 0, "Should be removed after grace period expires")
    }

    func testHitTestFindsCorrectInstance() {
        var tracker = InstanceTracker(stabilityDuration: 0.0, tapPadding: 0)
        let instanceA = DetectedInstance(id: 1, boundingBox: CGRect(x: 0.0, y: 0.0, width: 0.3, height: 0.3))
        let instanceB = DetectedInstance(id: 2, boundingBox: CGRect(x: 0.5, y: 0.5, width: 0.3, height: 0.3))

        tracker.update(instances: [instanceA, instanceB], timestamp: 0.0)

        let hit = tracker.instance(at: CGPoint(x: 0.15, y: 0.15))
        XCTAssertEqual(hit?.id, 1)

        let hitB = tracker.instance(at: CGPoint(x: 0.65, y: 0.65))
        XCTAssertEqual(hitB?.id, 2)

        let miss = tracker.instance(at: CGPoint(x: 0.9, y: 0.9))
        XCTAssertNil(miss)
    }

    // Tap padding: slightly outside the bounding box should still hit
    func testTapPaddingAllowsNearMisses() {
        var tracker = InstanceTracker(stabilityDuration: 0.0, tapPadding: 0.05)
        let instance = DetectedInstance(id: 1, boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.2, height: 0.2))
        // bbox goes from (0.2, 0.2) to (0.4, 0.4)

        tracker.update(instances: [instance], timestamp: 0.0)

        // Just outside the top edge (0.18 < 0.2 but within 0.05 padding)
        let nearMiss = tracker.instance(at: CGPoint(x: 0.3, y: 0.18))
        XCTAssertEqual(nearMiss?.id, 1, "Tap slightly outside bbox should hit with padding")

        // Just outside the right edge (0.43 > 0.4 but within 0.05 padding)
        let nearMissRight = tracker.instance(at: CGPoint(x: 0.43, y: 0.3))
        XCTAssertEqual(nearMissRight?.id, 1, "Tap slightly outside right edge should hit with padding")

        // Way outside — should miss even with padding
        let farMiss = tracker.instance(at: CGPoint(x: 0.8, y: 0.8))
        XCTAssertNil(farMiss, "Tap far outside should still miss")
    }

    func testTapPaddingClampsToNormalizedBounds() {
        var tracker = InstanceTracker(stabilityDuration: 0.0, tapPadding: 0.1)
        // Instance at the edge of the frame
        let instance = DetectedInstance(id: 1, boundingBox: CGRect(x: 0.0, y: 0.0, width: 0.1, height: 0.1))

        tracker.update(instances: [instance], timestamp: 0.0)

        // Tap at the origin should hit (within padded region)
        let hit = tracker.instance(at: CGPoint(x: 0.05, y: 0.05))
        XCTAssertEqual(hit?.id, 1)
    }

    func testIoUMatchingAcrossFrames() {
        var tracker = InstanceTracker(stabilityDuration: 0.5)
        tracker.update(instances: [
            DetectedInstance(id: 1, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2))
        ], timestamp: 0.0)
        tracker.update(instances: [
            DetectedInstance(id: 5, boundingBox: CGRect(x: 0.12, y: 0.12, width: 0.2, height: 0.2))
        ], timestamp: 0.3)
        tracker.update(instances: [
            DetectedInstance(id: 8, boundingBox: CGRect(x: 0.12, y: 0.12, width: 0.2, height: 0.2))
        ], timestamp: 0.5)

        let tappable = tracker.tappableInstances(at: 0.5)
        XCTAssertEqual(tappable.count, 1)
    }

    func testMultipleEmptyFramesDontDuplicateInstances() {
        var tracker = InstanceTracker(stabilityDuration: 0.3, gracePeriod: 2.0)
        let instance = DetectedInstance(id: 1, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2))

        tracker.update(instances: [instance], timestamp: 0.0)
        tracker.update(instances: [], timestamp: 0.1)
        tracker.update(instances: [], timestamp: 0.2)
        tracker.update(instances: [], timestamp: 0.3)

        // Should still have exactly 1 tracked instance, not duplicated
        XCTAssertEqual(tracker.trackedInstances.count, 1, "Grace period should not duplicate instances across empty frames")
    }
}
