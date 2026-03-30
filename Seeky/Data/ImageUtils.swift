import CoreImage
import Accelerate

/// Shared CIContext — expensive to create, safe to reuse across threads.
let sharedCIContext = CIContext()

/// Cosine similarity between two equal-length float vectors using vDSP.
func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return 0 }
    var dot: Float = 0
    var normA: Float = 0
    var normB: Float = 0
    vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
    vDSP_dotpr(a, 1, a, 1, &normA, vDSP_Length(a.count))
    vDSP_dotpr(b, 1, b, 1, &normB, vDSP_Length(b.count))
    guard normA > 0 && normB > 0 else { return 0 }
    return dot / (sqrt(normA) * sqrt(normB))
}

/// Resize a CVPixelBuffer to the target size using CIImage scaling.
func resizePixelBuffer(_ buffer: CVPixelBuffer, to size: CGSize) -> CVPixelBuffer? {
    let ciImage = CIImage(cvPixelBuffer: buffer)
    let scaleX = size.width / ciImage.extent.width
    let scaleY = size.height / ciImage.extent.height
    let resized = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

    var outputBuffer: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height),
                       kCVPixelFormatType_32BGRA, nil, &outputBuffer)
    guard let output = outputBuffer else { return nil }
    sharedCIContext.render(resized, to: output)
    return output
}
