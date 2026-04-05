<div align="center">

# wasm-component-examples

<sup>Working examples for WebAssembly Component Model</sup>

&nbsp;

![Bazel](https://img.shields.io/badge/Bazel-43A047?style=flat-square&logo=bazel&logoColor=white&labelColor=1a1b27)
![WebAssembly](https://img.shields.io/badge/WebAssembly-654FF0?style=flat-square&logo=webassembly&logoColor=white&labelColor=1a1b27)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square&labelColor=1a1b27)

</div>

&nbsp;

Examples demonstrating the [WebAssembly Component Model](https://component-model.bytecodealliance.org/) using [rules_wasm_component](https://github.com/pulseengine/rules_wasm_component) with Bazel.

> [!NOTE]
> Part of the PulseEngine toolchain. Demonstrates patterns used across the PulseEngine toolchain.

## Examples

| Example | Language | Type | Description |
|---------|----------|------|-------------|
| `//c:hello_c` | C | Library | Exports `greeter` interface |
| `//cpp:hello_cpp` | C++ | Library | Exports `greeter` interface |
| `//cpp:stats` | C++ | Library | Descriptive statistics (mean, median, percentile) |
| `//go:hello_go` | Go | CLI | Hello world with TinyGo |
| `//rust:hello_rust` | Rust | CLI | Hello world CLI component |
| `//rust:calculator` | Rust | CLI | Arithmetic calculator |
| `//rust:datetime` | Rust | CLI | Shows current date/time |
| `//rust:yolo_inference` | Rust | CLI + WASI-NN | YOLO object detection |
| `//rust_p3:text_processor` | Rust | Library (P3) | Async text analysis and transformation |
| `//rust_p3:concurrent_tasks` | Rust | Library (P3) | Async math computations (fibonacci, prime, collatz) |

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

## Project Structure

```
.
├── c/                    # C hello world component
├── cpp/                  # C++ hello world component
├── rust/                 # Rust components
│   ├── src/
│   │   ├── main.rs           # hello_rust
│   │   ├── calculator.rs     # calculator
│   │   ├── datetime.rs       # datetime
│   │   └── yolo_inference.rs # YOLO detection logic
│   └── wit/yolo.wit
├── rust_p3/              # Rust P3 async components
│   ├── src/
│   │   ├── text_processor.rs  # text analysis & transformation
│   │   └── concurrent.rs      # math computations (fib, prime, collatz)
│   └── wit/
│       ├── text_processor.wit
│       └── concurrent.wit
├── models/               # ONNX models (download separately)
├── MODULE.bazel
└── BUILD.bazel
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

Export `wasi:cli/run` for direct execution with Wasmtime.

## CI/CD

- **CI** (`ci.yml`): Builds all components on Linux and macOS, runs Rust CLI tests
- **Release** (`release.yml`): Creates signed releases with provenance attestation

## WASI Preview 3 (P3) Async Components

The `rust_p3/` directory demonstrates WASI Preview 3 async components. Setting `wasi_version = "p3"` on a `rust_wasm_component_bindgen` target passes `--async` to wit-bindgen, making all exported trait methods `async fn`. This enables cooperative concurrency within the Component Model.

### Text Processor (`//rust_p3:text_processor`)

Exports two async interfaces:
- **analyzer** — `analyze()` returns stats (word count, unique words, avg length), `word_frequencies()` returns sorted frequency table, `search_positions()` finds pattern occurrences
- **transformer** — `transform()` for case/reverse/title-case, `caesar_cipher()` for shift ciphers, `replace_all()` for substitution

### Concurrent Tasks (`//rust_p3:concurrent_tasks`)

Exports async mathematical functions designed for concurrent dispatch:
- `fibonacci(n)`, `factorial(n)`, `is_prime(n)`, `collatz_steps(n)`
- `compute_batch(numbers)` — runs fibonacci, prime check, and collatz on every input

```bash
# Build all P3 components
bazel build //rust_p3:all
```

See `rust_p3/src/text_processor.rs` and `rust_p3/src/concurrent.rs` for the async implementation pattern using `#[cfg(target_arch)]` gates.

## License

Apache-2.0

---

<div align="center">

<sub>Part of <a href="https://github.com/pulseengine">PulseEngine</a> &mdash; formally verified WebAssembly toolchain for safety-critical systems</sub>

</div>
