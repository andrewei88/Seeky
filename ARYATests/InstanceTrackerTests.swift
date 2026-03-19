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

    func testInstanceDisappearsAndResets() {
        var tracker = InstanceTracker(stabilityDuration: 0.5)
        let instance = DetectedInstance(id: 1, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2))

        tracker.update(instances: [instance], timestamp: 0.0)
        tracker.update(instances: [instance], timestamp: 0.3)
        tracker.update(instances: [], timestamp: 0.4)
        tracker.update(instances: [instance], timestamp: 0.5)

        let tappable = tracker.tappableInstances(at: 0.5)
        XCTAssertTrue(tappable.isEmpty)
    }

    func testHitTestFindsCorrectInstance() {
        var tracker = InstanceTracker(stabilityDuration: 0.0)
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
}
