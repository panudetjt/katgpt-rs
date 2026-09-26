//! Magika file type detector — pure Rust port, F32 weights, WASM/SIMD128 support.
//!
//! A self-contained Rust implementation of Google Magika (standard_v3_3) that
//! compiles to `wasm32-unknown-unknown` for browser use and runs natively for
//! tests and benchmarks. All 784,223 model parameters stay in full F32
//! precision (no quantization), so numerical output matches the ONNX reference
//! to within softmax tolerance ~2e-4.

pub mod content;
pub mod features;
pub mod inference;
pub mod model;
pub mod postprocess;
pub mod wasm;

/// Full pipeline: byte slice → features → forward → post-processed `FileType`.
pub fn identify_bytes(data: &[u8]) -> postprocess::FileType {
    match features::extract_features(data) {
        features::FeaturesOrRuled::Ruled(label) => postprocess::FileType {
            content_type: label,
            inferred_label: label,
            score: 1.0,
            overwrite_reason: None,
        },
        features::FeaturesOrRuled::Features(f) => {
            let scores = inference::forward(&f);
            postprocess::postprocess(&scores)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_empty() {
        let ft = identify_bytes(b"");
        assert_eq!(ft.content_type, "empty");
        assert_eq!(ft.score, 1.0);
    }

    #[test]
    fn tiny_text_is_txt() {
        let ft = identify_bytes(b"hi");
        assert_eq!(ft.content_type, "txt");
    }

    #[test]
    fn tiny_binary_is_unknown() {
        let ft = identify_bytes(&[0u8, 159, 146, 150]);
        assert_eq!(ft.content_type, "unknown");
    }
}
