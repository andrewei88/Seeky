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
    let gracePeriod: Double
    let tapPadding: CGFloat
    private let iouThreshold: Double
    private(set) var trackedInstances: [TrackedInstance] = []

    init(stabilityDuration: Double = 0.3, iouThreshold: Double = 0.3, gracePeriod: Double = 1.5, tapPadding: CGFloat = 0.04) {
        self.stabilityDuration = stabilityDuration
        self.iouThreshold = iouThreshold
        self.gracePeriod = gracePeriod
        self.tapPadding = tapPadding
    }

    mutating func update(instances: [DetectedInstance], timestamp: Double) {
        var matched = Set<Int>()
        var newTracked: [TrackedInstance] = []

        for instance in instances {
            if let matchIndex = bestMatch(for: instance) {
                var tracked = trackedInstances[matchIndex]
                tracked.currentInstance = instance
                tracked.lastSeenTimestamp = timestamp
                newTracked.append(tracked)
                matched.insert(matchIndex)
            } else {
                newTracked.append(TrackedInstance(
                    currentInstance: instance,
                    firstSeenTimestamp: timestamp,
                    lastSeenTimestamp: timestamp
                ))
            }
        }

        // Keep unmatched instances alive during grace period
        for (index, tracked) in trackedInstances.enumerated() {
            if !matched.contains(index) && (timestamp - tracked.lastSeenTimestamp) < gracePeriod {
                newTracked.append(tracked)
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
            .first { paddedBox($0.boundingBox).contains(point) }
    }

    private func paddedBox(_ rect: CGRect) -> CGRect {
        CGRect(
            x: max(0, rect.origin.x - tapPadding),
            y: max(0, rect.origin.y - tapPadding),
            width: min(1.0 - max(0, rect.origin.x - tapPadding), rect.width + tapPadding * 2),
            height: min(1.0 - max(0, rect.origin.y - tapPadding), rect.height + tapPadding * 2)
        )
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
