//! WebAssembly bindings via `wasm-bindgen` for browser and Node.js environments.

use wasm_bindgen::prelude::*;

use crate::{identify_bytes, postprocess::type_info_of};

/// Result of identifying a file with Magika.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct MagikaResult {
    label: String,
    mime_type: String,
    group: String,
    description: String,
    extensions: Vec<String>,
    is_text: bool,
    score: f32,
}

#[wasm_bindgen]
impl MagikaResult {
    /// Canonical content type label (e.g. "pdf", "python", "png", "txt", "unknown").
    #[wasm_bindgen(getter)]
    pub fn label(&self) -> String {
        self.label.clone()
    }

    /// Official MIME type (e.g. "application/pdf", "image/png").
    #[wasm_bindgen(getter)]
    pub fn mime_type(&self) -> String {
        self.mime_type.clone()
    }

    /// High-level content group (e.g. "document", "image", "code", "archive").
    #[wasm_bindgen(getter)]
    pub fn group(&self) -> String {
        self.group.clone()
    }

    /// Human-readable description (e.g. "Portable Document Format").
    #[wasm_bindgen(getter)]
    pub fn description(&self) -> String {
        self.description.clone()
    }

    /// Common file extensions as a comma-separated string (e.g. "pdf" or "jpg,jpeg").
    #[wasm_bindgen(getter)]
    pub fn extensions(&self) -> String {
        self.extensions.join(",")
    }

    /// True if content is text-based.
    #[wasm_bindgen(getter)]
    pub fn is_text(&self) -> bool {
        self.is_text
    }

    /// Model prediction score in `[0.0, 1.0]`.
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Export result as a JSON string.
    pub fn to_json(&self) -> String {
        let exts = self
            .extensions
            .iter()
            .map(|s| format!("\"{}\"", s))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"label\":\"{}\",\"mime_type\":\"{}\",\"group\":\"{}\",\"description\":\"{}\",\"extensions\":[{}],\"is_text\":{},\"score\":{:.6}}}",
            self.label,
            self.mime_type,
            self.group,
            self.description.replace('"', "\\\""),
            exts,
            self.is_text,
            self.score
        )
    }
}

/// Identifies a file's content type from a Uint8Array byte slice.
#[wasm_bindgen]
pub fn identify(bytes: &[u8]) -> MagikaResult {
    let ft = identify_bytes(bytes);
    let info = type_info_of(&ft);
    MagikaResult {
        label: ft.content_type.to_string(),
        mime_type: info.mime_type.to_string(),
        group: info.group.to_string(),
        description: info.description.to_string(),
        extensions: info.extensions.iter().map(|s| s.to_string()).collect(),
        is_text: info.is_text,
        score: ft.score,
    }
}

/// Identifies a file and directly returns a JSON string representation.
#[wasm_bindgen]
pub fn identify_json(bytes: &[u8]) -> String {
    identify(bytes).to_json()
}
