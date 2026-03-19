import Foundation
import CoreGraphics

struct DetectedInstance: Equatable {
    let id: Int
    let boundingBox: CGRect // In normalized image coordinates (0-1)
}

struct TrackedInstance {
    var currentInstance: DetectedInstance
    var firstSeenTimestamp: Double
    var lastSeenTimestamp: Double
}

struct InstanceTracker {
    let stabilityDuration: Double
    private let iouThreshold: Double
    private(set) var trackedInstances: [TrackedInstance] = []

    init(stabilityDuration: Double = 0.5, iouThreshold: Double = 0.3) {
        self.stabilityDuration = stabilityDuration
        self.iouThreshold = iouThreshold
    }

    mutating func update(instances: [DetectedInstance], timestamp: Double) {
        var newTracked: [TrackedInstance] = []

        for instance in instances {
            if let matchIndex = bestMatch(for: instance) {
                // Existing instance — update
                var tracked = trackedInstances[matchIndex]
                tracked.currentInstance = instance
                tracked.lastSeenTimestamp = timestamp
                newTracked.append(tracked)
            } else {
                // New instance
                newTracked.append(TrackedInstance(
                    currentInstance: instance,
                    firstSeenTimestamp: timestamp,
                    lastSeenTimestamp: timestamp
                ))
            }
        }

        trackedInstances = newTracked
    }

    func tappableInstances(at currentTime: Double) -> [DetectedInstance] {
        trackedInstances
            .filter { (currentTime - $0.firstSeenTimestamp) >= stabilityDuration }
            .map(\.currentInstance)
    }

    func instance(at point: CGPoint) -> DetectedInstance? {
        trackedInstances
            .map(\.currentInstance)
            .first { $0.boundingBox.contains(point) }
    }

    private func bestMatch(for instance: DetectedInstance) -> Int? {
        var bestIndex: Int?
        var bestIoU: CGFloat = 0

        for (index, tracked) in trackedInstances.enumerated() {
            let iou = computeIoU(tracked.currentInstance.boundingBox, instance.boundingBox)
            if iou > CGFloat(iouThreshold) && iou > bestIoU {
                bestIoU = iou
                bestIndex = index
            }
        }

        return bestIndex
    }

    private func computeIoU(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        guard !intersection.isNull else { return 0 }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = (a.width * a.height) + (b.width * b.height) - intersectionArea
        guard unionArea > 0 else { return 0 }
        return intersectionArea / unionArea
    }
}
