import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager

    func makeUIView(context: Context) -> CameraUIView {
        let view = CameraUIView()
        view.previewLayer = cameraManager.previewLayer
        return view
    }

    func updateUIView(_ uiView: CameraUIView, context: Context) {
        uiView.updateLayout()
    }
}

class CameraUIView: UIView {
    var previewLayer: AVCaptureVideoPreviewLayer? {
        didSet {
            oldValue?.removeFromSuperlayer()
            if let layer = previewLayer {
                self.layer.addSublayer(layer)
                layer.frame = bounds
            }
        }
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }

    func updateLayout() {
        previewLayer?.frame = bounds
    }
}
