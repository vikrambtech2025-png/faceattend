# Model Card

## Detection + Recognition
- Model: insightface `buffalo_l` (RetinaFace detection + ArcFace R100 recognition)
- Embedding: 512-d, L2-normalized
- License: non-commercial research (insightface). For commercial use, swap to a commercially licensed ArcFace ONNX model.

## Liveness
- Active: MediaPipe FaceMesh + EAR blink detection
- Passive: Laplacian variance + color saturation heuristic
- Limitations: heuristic, not production-grade; recommend MiniFASNet for real deployments.

## Known limitations
- Performance varies across demographics (not yet measured).
- Sensitive to lighting, occlusion, extreme poses.
- Spoof detection weak against high-quality 3D masks / deepfakes.