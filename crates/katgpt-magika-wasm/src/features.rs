//! Feature extraction matching Google Magika's preprocessing specification.

use crate::model::{BEG_SIZE, BLOCK_SIZE, FEATURES_SIZE, MIN_FILE_SIZE_FOR_DL, PADDING_TOKEN};

/// Extracted features or rule-based result.
#[derive(Debug, Clone)]
pub enum FeaturesOrRuled {
    /// 2048 i32 tokens (0..255 byte values, 256 for padding).
    Features(Vec<i32>),
    /// Early rule-based classification (empty, txt, or unknown).
    Ruled(&'static str),
}

/// Extracts features from byte slice.
pub fn extract_features(data: &[u8]) -> FeaturesOrRuled {
    let file_len = data.len();
    if file_len == 0 {
        return FeaturesOrRuled::Ruled("empty");
    }

    let buffer_size = std::cmp::min(BLOCK_SIZE, file_len);
    let content_beg = &data[..buffer_size];
    let beg = strip_prefix(content_beg);

    let content_end = if file_len == buffer_size {
        content_beg
    } else {
        &data[file_len - buffer_size..]
    };
    let end = strip_suffix(content_end);

    let mut features = vec![PADDING_TOKEN; FEATURES_SIZE];
    // Beg is left-aligned in features[..BEG_SIZE]
    copy_features(&mut features[..BEG_SIZE], beg, 0);
    // End is right-aligned in features[BEG_SIZE..]
    copy_features(&mut features[BEG_SIZE..], end, 1);

    // Rule-based check: if file has fewer than MIN_FILE_SIZE_FOR_DL (8) non-padding characters
    if features[MIN_FILE_SIZE_FOR_DL - 1] == PADDING_TOKEN {
        let content_type = match std::str::from_utf8(content_beg) {
            Ok(_) => "txt",
            Err(_) => "unknown",
        };
        return FeaturesOrRuled::Ruled(content_type);
    }

    FeaturesOrRuled::Features(features)
}

fn copy_features(dst: &mut [i32], src: &[u8], align: usize) {
    let len = std::cmp::min(dst.len(), src.len());
    let dst_len = dst.len();
    let dst = &mut dst[(dst_len - len) * align..][..len];
    let src = &src[(src.len() - len) * align..][..len];
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s as i32;
    }
}

fn strip_prefix(xs: &[u8]) -> &[u8] {
    let mut start = 0;
    while start < xs.len() && is_whitespace(xs[start]) {
        start += 1;
    }
    &xs[start..]
}

fn strip_suffix(xs: &[u8]) -> &[u8] {
    let mut end = xs.len();
    while end > 0 && is_whitespace(xs[end - 1]) {
        end -= 1;
    }
    &xs[..end]
}

#[inline(always)]
fn is_whitespace(x: u8) -> bool {
    x.is_ascii_whitespace() || x == 0x0b
}
