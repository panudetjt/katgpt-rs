//! Magika model weights and forward pass (F32, no quantization).
//!
//! Weights are raw f32 tensors extracted from Google's official ONNX model
//! (standard_v3_3, 784,223 parameters total, 3 MB binary). Packed into a
//! single `weights.bin` file loaded via `include_bytes!`.
//!
//! Layout:
//!   magic "MAGIKAF3" + u32 version 2
//!   emb_w (257, 64), emb_b (64), ln0_gamma (512), ln0_beta (512),
//!   conv_w (512, 5, 256), conv_b (512), ln1_gamma (512), ln1_beta (512),
//!   dense1_w (214, 512), dense1_b (214)

pub const BEG_SIZE: usize = 1024;
pub const END_SIZE: usize = 1024;
pub const BLOCK_SIZE: usize = 4096;
pub const PADDING_TOKEN: i32 = 256;
pub const MIN_FILE_SIZE_FOR_DL: usize = 8;
pub const FEATURES_SIZE: usize = BEG_SIZE + END_SIZE;

pub const EMB_ROWS: usize = 257;
pub const EMB_DIM: usize = 64;
pub const HIDDEN: usize = 512;
pub const TIME: usize = 256;
pub const KERNEL: usize = 5;
pub const CONV_OUT: usize = HIDDEN - KERNEL + 1; // 508
pub const NUM_LABELS: usize = 214;

/// Raw f32 weights — 3 MB blob, no quantization (per user instruction).
pub static WEIGHTS_BYTES: &[u8] = include_bytes!("../weights.bin");

/// Model weight tensors viewed as f32 slices.
pub struct Weights {
    pub emb_w: &'static [f32],      // [257 * 64]
    pub emb_b: &'static [f32],      // [64]
    pub ln0_gamma: &'static [f32],  // [512]
    pub ln0_beta: &'static [f32],   // [512]
    pub conv_w: &'static [f32],     // [512 * 5 * 256], layout [out_c][k][in_c]
    pub conv_b: &'static [f32],     // [512]
    pub ln1_gamma: &'static [f32],  // [512]
    pub ln1_beta: &'static [f32],   // [512]
    pub dense1_w: &'static [f32],   // [214 * 512], layout [label][hidden]
    pub dense1_b: &'static [f32],   // [214]
}

/// Loads and validates weights from the embedded binary blob.
pub fn get_weights() -> &'static Weights {
    static CELL: std::sync::OnceLock<Weights> = std::sync::OnceLock::new();
    CELL.get_or_init(|| {
        assert!(WEIGHTS_BYTES.len() > 12, "weights.bin truncated");
        assert!(&WEIGHTS_BYTES[0..8] == b"MAGIKAF3", "weights.bin bad magic");
        let version = u32::from_le_bytes(WEIGHTS_BYTES[8..12].try_into().unwrap());
        assert!(version == 2, "weights.bin unsupported version {version}");

        let mut off = 12;
        let mut take = |n: usize| -> &'static [f32] {
            let bytes = &WEIGHTS_BYTES[off..off + n * 4];
            off += n * 4;
            let mut out = Vec::with_capacity(n);
            let (chunks, _) = bytes.as_chunks::<4>();
            for chunk in chunks {
                out.push(f32::from_le_bytes(*chunk));
            }
            Box::leak(out.into_boxed_slice())
        };

        let emb_w = take(EMB_ROWS * EMB_DIM);
        let emb_b = take(EMB_DIM);
        let ln0_gamma = take(HIDDEN);
        let ln0_beta = take(HIDDEN);
        let conv_w = take(HIDDEN * KERNEL * TIME);
        let conv_b = take(HIDDEN);
        let ln1_gamma = take(HIDDEN);
        let ln1_beta = take(HIDDEN);
        let dense1_w = take(NUM_LABELS * HIDDEN);
        let dense1_b = take(NUM_LABELS);
        assert!(off == WEIGHTS_BYTES.len(), "weights.bin has trailing bytes");

        Weights {
            emb_w,
            emb_b,
            ln0_gamma,
            ln0_beta,
            conv_w,
            conv_b,
            ln1_gamma,
            ln1_beta,
            dense1_w,
            dense1_b,
        }
    })
}
