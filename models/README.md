# Models Directory

This directory contains ONNX models for the YOLO inference example.

## Download Models

Download YOLOv8 Nano (12.8 MB):

```bash
curl -L -o models/yolov8n.onnx \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx

# Create subdirectory for wasmtime (expects model.onnx inside named directory)
mkdir -p models/yolov8n
ln -sf ../yolov8n.onnx models/yolov8n/model.onnx
```

## Available Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| YOLOv8n | 12.8 MB | Fastest | Good |
| YOLOv8s | 44.4 MB | Fast | Better |
| YOLOv8m | 103 MB | Medium | High |
| YOLOv8l | 175 MB | Slow | Higher |
| YOLOv8x | 268 MB | Slowest | Highest |

For this example, YOLOv8n is recommended as it provides the best balance of speed and size for WASM inference.
