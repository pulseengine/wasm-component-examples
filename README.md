# WebAssembly Component Model Examples

[![CI](https://github.com/pulseengine/wasm-component-examples/actions/workflows/ci.yml/badge.svg)](https://github.com/pulseengine/wasm-component-examples/actions/workflows/ci.yml)
[![Integration Test](https://github.com/pulseengine/wasm-component-examples/actions/workflows/integration-test.yml/badge.svg)](https://github.com/pulseengine/wasm-component-examples/actions/workflows/integration-test.yml)

Examples demonstrating the [WebAssembly Component Model](https://component-model.bytecodealliance.org/) using [rules_wasm_component](https://github.com/pulseengine/rules_wasm_component) with Bazel.

## Examples

| Example | Language | Type | Profiles | Description |
|---------|----------|------|----------|-------------|
| `//c:hello_c` | C | Library | debug, release | Exports `greeter` interface |
| `//cpp:hello_cpp` | C++ | Library | debug, release | Exports `greeter` interface |
| `//go:hello_go` | Go | CLI | release | Hello world with TinyGo |
| `//rust:hello_rust` | Rust | CLI | - | Hello world CLI component |
| `//rust:calculator` | Rust | CLI | - | Arithmetic calculator (`8 + 8`) |
| `//rust:datetime` | Rust | CLI | - | Shows current date/time |
| `//rust:yolo_inference` | Rust | CLI + WASI-NN | debug, release | YOLO object detection |

## Prerequisites

- [Bazel](https://bazel.build/) 7.0+
- [Wasmtime](https://wasmtime.dev/) (for running components)
- For YOLO: Wasmtime compiled with ONNX backend support

## Build

```bash
# Build all examples
bazel build //...

# Build specific example
bazel build //rust:hello_rust
bazel build //rust:yolo_inference_release
```

## Run

### Rust CLI Components

```bash
# Hello world
wasmtime bazel-bin/rust/hello_rust.runfiles/_main/rust/hello_rust_host.wasm

# Calculator
wasmtime bazel-bin/rust/calculator.runfiles/_main/rust/calculator_host.wasm 8 + 8

# DateTime
wasmtime bazel-bin/rust/datetime.runfiles/_main/rust/datetime_host.wasm
```

### YOLO Object Detection

Requires ONNX model (see `models/README.md`):

```bash
# Download model
curl -L -o models/yolov8n.onnx \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx
mkdir -p models/yolov8n && ln -sf ../yolov8n.onnx models/yolov8n/model.onnx

# Run detection
wasmtime run --dir . -S cli -S nn -S nn-graph=onnx::./models/yolov8n \
  bazel-out/darwin_arm64-fastbuild-ST-*/bin/rust/yolo_inference_wasm_lib_release_wasm_base.wasm \
  ./bus.jpg
```

Output:
```
YOLO Object Detection Demo
---------------------------
[1] Loading image...
[2] Loading model...
[3] Preprocessing image...
[4] Creating execution context...
[5] Running inference...
[6] Reading output...
[7] Post-processing...

======================================================
  Results: 5 detections
======================================================
  [ 1] person               87.2% @ pixel (...)
  [ 2] bus                  85.1% @ pixel (...)
```

## Project Structure

```
.
├── c/                    # C hello world component
│   ├── src/hello.c
│   └── wit/hello.wit
├── cpp/                  # C++ hello world component
│   ├── src/hello.cpp
│   └── wit/hello.wit
├── rust/                 # Rust components
│   ├── src/
│   │   ├── main.rs           # hello_rust
│   │   ├── calculator.rs     # calculator
│   │   ├── datetime.rs       # datetime
│   │   ├── lib.rs            # yolo component entry
│   │   └── yolo_inference.rs # YOLO detection logic
│   └── wit/yolo.wit
├── models/               # ONNX models (download separately)
├── MODULE.bazel          # Bazel module definition
└── BUILD.bazel           # Root build file
```

## Component Types

### Library Components (C/C++)

Export custom interfaces that can be composed with other components:

```wit
interface greeter {
    greet: func() -> string;
}

world hello-c {
    export greeter;
}
```

### CLI Components (Rust)

Export `wasi:cli/run` for direct execution with wasmtime:

```wit
world yolo-detection {
    import wasi:nn/graph@0.2.0-rc-2024-10-28;
    import wasi:nn/tensor@0.2.0-rc-2024-10-28;
    import wasi:nn/inference@0.2.0-rc-2024-10-28;
    // ... other imports
    export wasi:cli/run@0.2.3;
}
```

## CI/CD

This repository uses GitHub Actions for:

- **CI** (`ci.yml`): Builds all components on Linux and macOS, runs Rust CLI tests
- **Release** (`release.yml`): Creates signed releases with provenance attestation

### Creating a Release

```bash
# Tag and push to create a release
git tag v1.0.0
git push origin v1.0.0
```

The release workflow will:
1. Build all components
2. Sign them with wasmsign2
3. Create a GitHub release with artifacts
4. Generate provenance attestation

## License

Apache-2.0
