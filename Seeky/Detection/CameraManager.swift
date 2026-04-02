import AVFoundation
import UIKit

protocol CameraManagerDelegate: AnyObject {
    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime)
}

enum CameraStatus {
    case notDetermined
    case authorized
    case denied
}

final class CameraManager: NSObject, ObservableObject {
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.seeky.camera.session")
    private let outputQueue = DispatchQueue(label: "com.seeky.camera.output")

    weak var delegate: CameraManagerDelegate?
    @Published var status: CameraStatus = .notDetermined
    private var frameCount = 0
    private lazy var _previewLayer: AVCaptureVideoPreviewLayer = {
        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        layer.videoGravity = .resizeAspectFill
        return layer
    }()

    var previewLayer: AVCaptureVideoPreviewLayer {
        _previewLayer
    }

    /// Convert a screen point (in the preview layer's coordinate system) to
    /// normalized image coordinates (0-1), accounting for resizeAspectFill cropping.
    func imagePoint(fromScreenPoint screenPoint: CGPoint) -> CGPoint {
        previewLayer.captureDevicePointConverted(fromLayerPoint: screenPoint)
    }

    func configure() {
        sessionQueue.async { [weak self] in
            self?.checkAuthorizationAndSetup()
        }
    }

    func start() {
        sessionQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
        }
    }

    private func checkAuthorizationAndSetup() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupSession()
            DispatchQueue.main.async { self.status = .authorized }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    self?.sessionQueue.async { self?.setupSession() }
                    DispatchQueue.main.async { self?.status = .authorized }
                } else {
                    DispatchQueue.main.async { self?.status = .denied }
                }
            }
        default:
            DispatchQueue.main.async { self.status = .denied }
        }
    }

    private func setupSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1920x1080

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera),
              captureSession.canAddInput(input) else {
            captureSession.commitConfiguration()
            return
        }

        captureSession.addInput(input)

        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        guard captureSession.canAddOutput(videoOutput) else {
            captureSession.commitConfiguration()
            return
        }

        captureSession.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            connection.videoRotationAngle = 90 // Portrait
        }

        captureSession.commitConfiguration()
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameCount += 1
        // Process every 3rd frame for segmentation (~10fps on 30fps feed)
        guard frameCount % 3 == 0 else { return }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        delegate?.cameraManager(self, didOutput: pixelBuffer, timestamp: timestamp)
    }
}
