# katgpt-magika-wasm

Pure Rust WebAssembly port of [Google Magika](https://github.com/google/magika) deep learning file type detector. Runs in the browser and Node.js with **F32 weights (no quantization)** and **WASM SIMD128** hardware acceleration.

---

## Highlights

- **Modelless inference architecture in pure Rust**: No Python runtime, no ONNX Runtime dependency, no C++ bindings.
- **Zero quantization loss**: Model weights remain in 32-bit floating point (`F32`), matching official Google Magika `standard_v3_3` model outputs to numerical precision (max absolute difference ~`2e-6`).
- **SIMD128 WebAssembly acceleration**: Uses WebAssembly SIMD128 vector intrinsics (`f32x4_add`, `f32x4_mul`, `v128_load`) for parallel dot products, falling back to 4-lane unrolled loops on non-SIMD platforms.
- **Compact bundle**: Entire WebAssembly binary including all 784,223 F32 weights and 214 content type metadata tables is only **3.1 MB**.
- **Browser & Node.js ready**: Exported via `wasm-bindgen` with ESM modules and TypeScript definitions (`.d.ts`).

---

## Model & Architecture

| Parameter | Value |
|---|---|
| Model | Google Magika `standard_v3_3` |
| Parameters | 784,223 parameters (~3.13 MB in F32) |
| Target Labels | 214 content types (plus 6 rule-based specials: empty, txt, unknown, undefined, directory, symlink) |
| Sequence Length | 512 tokens (beg 256 + end 256) |
| Feature Extractor | Whitespace-stripped prefix & suffix alignment with padding token 256 |
| Architecture | Embedding (257x64) -> GELU -> LN0 -> Conv1D (k=5, 256->512) -> GELU -> GlobalMaxPool (508->1) -> LN1 -> Dense (512->214) -> Softmax |

---

## Directory Structure

```text
crates/katgpt-magika-wasm/
├── Cargo.toml
├── weights.bin              # Serialized F32 model weights (3.13 MB)
├── src/
│   ├── lib.rs               # Main library entry point
│   ├── model.rs             # Model constants and static weight loader
│   ├── content.rs           # 214 content types, thresholds, overwrite map & MIME metadata
│   ├── features.rs          # Feature extractor & short-file rule engine
│   ├── inference.rs         # Neural forward pass with SIMD128 / 4-lane fallback
│   ├── postprocess.rs       # Threshold calibration, argmax & overwrite mapper
│   └── wasm.rs              # wasm-bindgen JS bindings & MagikaResult
├── tests/
│   ├── test_parity.rs       # Real-world files parity test suite (PDF, PNG, Rust, HTML, Python, etc.)
│   └── test_ort_parity.rs   # Onnxruntime reference fixtures test suite
├── pkg/                     # wasm-bindgen generated JS/TS packages
└── demo/                    # Standalone web demo (drag & drop UI)
    ├── index.html
    └── pkg/                 # WASM bundle for the web demo
```

---

## Usage

### 1. In Rust

Add to your `Cargo.toml`:
```toml
[dependencies]
katgpt-magika-wasm = { path = "path/to/katgpt-magika-wasm" }
```

```rust
use katgpt_magika_wasm::{identify_bytes, postprocess::type_info_of};

let data = b"%PDF-1.4\n...";
let result = identify_bytes(data);

println!("Detected label: {}", result.content_type); // "pdf"
println!("Confidence: {:.2}%", result.score * 100.0); // 99.9%

let info = type_info_of(&result);
println!("MIME type: {}", info.mime_type);           // "application/pdf"
println!("Group: {}", info.group);                   // "document"
println!("Description: {}", info.description);       // "PDF document"
```

### 2. In JavaScript / TypeScript (Browser or Node.js)

```javascript
import init, { identify, identify_json } from './pkg/katgpt_magika_wasm.js';

// Initialize WASM module
await init();

// Identify a Uint8Array
const fileBytes = new Uint8Array(...);
const result = identify(fileBytes);

console.log(result.label);       // "python"
console.log(result.mime_type);   // "text/x-python"
console.log(result.group);       // "code"
console.log(result.description); // "Python source"
console.log(result.score);       // 0.9998
console.log(result.is_text);     // true

// Or export as JSON
console.log(result.to_json());
```

---

## Running the Web Demo

To run the interactive drag-and-drop web demo:

```bash
cd crates/katgpt-magika-wasm/demo
# Any static server works:
python3 -m http.server 8080
# or: bun x serve .
# or: npx serve .
```

Open `http://localhost:8080` in any modern web browser. Drop any file to see instantaneous type classification computed entirely client-side via WASM SIMD128.

---

## Building from Source

To compile the WebAssembly binary and regenerate bindings:

```bash
# 1. Build release WASM with SIMD128 enabled:
RUSTFLAGS="-C target-feature=+simd128" cargo build --release -p katgpt-magika-wasm --target wasm32-unknown-unknown

# 2. Generate web bindings with wasm-bindgen:
wasm-bindgen --target web --out-dir pkg ../../target/wasm32-unknown-unknown/release/katgpt_magika_wasm.wasm
```

---

## Verification & Parity

To run all unit and integration tests verifying parity with Google Magika:

```bash
cargo test -p katgpt-magika-wasm
cargo clippy -p katgpt-magika-wasm --all-targets
```
