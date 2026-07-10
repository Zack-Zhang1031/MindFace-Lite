# Interview Notes

## Short Project Pitch

MindFace-Lite is a lightweight speech-driven mouth animation project. It starts with a rule-based RMS energy demo, then adds a PyTorch training pipeline, MLP/LSTM/TCN/Transformer models, ONNX deployment, realtime queue simulation, and C++ control examples for UDP or serial actuator output.

## What I Can Explain

- Why RMS is a useful first baseline for mouth opening.
- How audio is split into video-rate frames.
- How the Dataset and DataLoader feed model training.
- Why MLP is a baseline and sequence models improve temporal context.
- How PyTorch checkpoint inference differs from ONNXRuntime inference.
- How to debug tensor shapes and normalization differences.
- How queues reduce blocking in realtime systems.
- How C++ fits into robot or embedded control.
- What RK3588/RKNN deployment requires beyond model export.

## Strong Answer Pattern

1. Start with the simplest working rule demo.
2. Log every intermediate output.
3. Replace the rule with a trained model only after the pipeline is validated.
4. Export and benchmark before claiming deployability.
5. Discuss edge constraints honestly: operators, quantization, latency, heat, DDR, and interfaces.

## Honest Boundary

This project is a learning and engineering demonstration. Production lip-sync requires real labeled audio-video data, accurate mouth landmarks or blendshape labels, better acoustic features, and user studies for visual quality.
