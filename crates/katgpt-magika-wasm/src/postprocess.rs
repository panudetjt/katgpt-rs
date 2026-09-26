//! Post-processing logic matching Google Magika — thresholds, overwrite map,
//! and canonical content-type resolution.

use crate::content::{get_overwrite_label, get_type_info_by_label, THRESHOLDS, TARGET_LABELS, TARGET_TYPE_INFOS};

/// Result of a single Magika inference.
#[derive(Debug, Clone, PartialEq)]
pub struct FileType {
    /// Final content type label after post-processing.
    pub content_type: &'static str,
    /// Raw neural-network inferred label.
    pub inferred_label: &'static str,
    /// Softmax score of the top prediction (0..1).
    pub score: f32,
    /// Reason the final content type differs from the inference (if any).
    pub overwrite_reason: Option<OverwriteReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverwriteReason {
    /// Score was below the per-type threshold; replaced with generic label.
    LowConfidence,
    /// Inference label is non-canonical and was remapped (e.g. randombytes → unknown).
    OverwriteMap,
}

/// Post-processes raw 214 softmax probabilities into a final `FileType`.
pub fn postprocess(scores: &[f32]) -> FileType {
    debug_assert_eq!(scores.len(), TARGET_LABELS.len());

    // Argmax
    let mut best = 0usize;
    for (i, &s) in scores.iter().enumerate() {
        if s > scores[best] {
            best = i;
        }
    }
    let score = scores[best];
    let inferred_label = TARGET_LABELS[best];

    // Low-confidence check: if below threshold, replace with generic label
    let threshold = THRESHOLDS[best];
    if score < threshold {
        let info = &TARGET_TYPE_INFOS[best];
        let fallback = if info.is_text { "txt" } else { "unknown" };
        return FileType {
            content_type: fallback,
            inferred_label,
            score,
            overwrite_reason: Some(OverwriteReason::LowConfidence),
        };
    }

    // Overwrite map (non-canonical labels)
    if let Some(overwritten_label) = get_overwrite_label(best) {
        return FileType {
            content_type: overwritten_label,
            inferred_label,
            score,
            overwrite_reason: Some(OverwriteReason::OverwriteMap),
        };
    }

    FileType {
        content_type: inferred_label,
        inferred_label,
        score,
        overwrite_reason: None,
    }
}

/// Helper: returns the `&'static TypeInfo` for a `FileType` result.
pub fn type_info_of(ft: &FileType) -> &'static crate::content::TypeInfo {
    get_type_info_by_label(ft.content_type)
}
