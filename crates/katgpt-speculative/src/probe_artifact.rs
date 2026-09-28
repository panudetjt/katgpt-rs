//! Weak-side probe artifact (Issue 865 T2) — the freeze/thaw wire for the
//! `probe_guidance` decode lane.
//!
//! The autoguidance weak side (Research 68 §7.2 / arXiv:2609.19356) is a cheap
//! probe read from an early-layer hidden state: a matched-width MLP connector
//! (the [`LatentDynamicsMLP`] class, Plan 217 — reused, not re-invented)
//! followed by the trunk's own output head, producing weak logits the strong
//! side extrapolates against. This module owns the ARTIFACT: a versioned,
//! BLAKE3-committed binary envelope around
//!
//! - the connector MLP ([`LatentDynamicsMLP`]),
//! - the shared `lm_head` (`[vocab * n_embd]`, copied from the trunk — the
//!   paper's matched-connector law: weak and strong share the unembedding),
//! - the tap-point metadata (`tap_layer`) the artifact was trained against.
//!
//! The commitment is computed over every byte preceding it and re-verified on
//! load (`CommitmentMismatch` on any tamper or truncation), consistent with
//! the modelless consumption rule: the trained artifact swaps in as a frozen
//! snapshot, never mutated at runtime. Training that produces this artifact
//! lives in the riir-train `nextlat_*` lane pattern (Issue 865 T2, the
//! katgpt-rs non-goal: no training pipeline here).
//!
//! Wire format v1 (little-endian throughout):
//!
//! ```text
//! [0..4)    magic "NLPA"
//! [4..8)    u32 format version (= 1)
//! [8..16)   u64 artifact version (monotonic, caller-managed)
//! [16..20)  u32 tap_layer
//! [20..24)  u32 n_embd
//! [24..28)  u32 vocab
//! [28..)    f32 arrays in order: mlp.norm_weight, mlp.norm_bias,
//!           mlp.fc1_weight, mlp.fc1_bias, mlp.fc2_weight, mlp.fc2_bias,
//!           mlp.fc3_weight, mlp.fc3_bias, lm_head
//! [..+32)   BLAKE3 over ALL preceding bytes (magic + header + weights)
//! ```

use std::path::Path;

use crate::belief_drafter::LatentDynamicsMLP;

/// Wire magic — "NLPA" (`NextLat` Probe Artifact).
const MAGIC: [u8; 4] = *b"NLPA";

/// Wire format version. Bump on any layout change; loaders reject other
/// versions rather than guessing.
const FORMAT_VERSION: u32 = 1;

/// Fixed header size: magic(4) + format version(4) + artifact version(8)
/// + `tap_layer(4`) + `n_embd(4`) + vocab(4).
const HEADER_LEN: usize = 28;

/// BLAKE3 commitment width.
const COMMITMENT_LEN: usize = 32;

/// Errors loading or building a [`ProbeArtifact`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ProbeArtifactError {
    /// File/object does not start with the `NLPA` magic.
    BadMagic([u8; 4]),
    /// Wire format version other than the one this loader speaks.
    UnsupportedFormatVersion(u32),
    /// BLAKE3 commitment over the payload does not match — the artifact was
    /// tampered with or truncated/corrupted in transit.
    CommitmentMismatch,
    /// A fixed-size region ended early.
    Truncated {
        /// What was being read when the bytes ran out.
        field: &'static str,
        /// Bytes the field requires.
        expected: usize,
        /// Bytes actually available.
        got: usize,
    },
    /// Structurally invalid content: zero dims, a weight array whose length
    /// disagrees with the header, or a tap layer the consuming kernel does
    /// not provide.
    InvalidShape(String),
    /// Filesystem failure.
    Io(String),
}

impl std::fmt::Display for ProbeArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic(m) => write!(f, "bad magic: expected {MAGIC:?}, got {m:?}"),
            Self::UnsupportedFormatVersion(v) => {
                write!(
                    f,
                    "unsupported format version: {v} (expected {FORMAT_VERSION})"
                )
            }
            Self::CommitmentMismatch => {
                write!(
                    f,
                    "BLAKE3 commitment mismatch — artifact tampered or corrupted"
                )
            }
            Self::Truncated {
                field,
                expected,
                got,
            } => {
                write!(f, "truncated {field}: expected {expected} bytes, got {got}")
            }
            Self::InvalidShape(msg) => write!(f, "invalid shape: {msg}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl std::error::Error for ProbeArtifactError {}

/// A frozen weak-side probe: connector MLP + shared trunk head + tap metadata,
/// sealed by a BLAKE3 commitment over the whole payload.
#[derive(Clone, Debug)]
pub struct ProbeArtifact {
    /// The connector MLP (the [`LatentDynamicsMLP`] class — Plan 217).
    pub mlp: LatentDynamicsMLP,
    /// Shared output head from the trunk, `[vocab * n_embd]` row-major.
    pub lm_head: Vec<f32>,
    /// Vocabulary size (`lm_head.len() == vocab * mlp.n_embd`).
    pub vocab: usize,
    /// The early-layer tap point this probe was trained against. The current
    /// D2F kernel (single-layer mini trunk) provides exactly one tap — layer
    /// 0, the pre-layer input residual — and the consuming side rejects
    /// artifacts declaring any deeper tap rather than silently misreading.
    pub tap_layer: usize,
    /// Caller-managed monotonic version (the freeze/thaw generation ordinal).
    pub version: u64,
    /// BLAKE3 commitment over every payload byte (magic + header + weights).
    pub blake3: [u8; 32],
}

impl ProbeArtifact {
    /// Seal an artifact from its parts, computing the BLAKE3 commitment.
    ///
    /// Validates the shape up front: `vocab > 0` and `lm_head` exactly
    /// `vocab * mlp.n_embd`. `tap_layer` is metadata only here (the consumer
    /// validates it against the kernel's capability at probe construction).
    pub fn from_parts(
        mlp: LatentDynamicsMLP,
        lm_head: Vec<f32>,
        tap_layer: usize,
        version: u64,
    ) -> Result<Self, ProbeArtifactError> {
        let n = mlp.n_embd;
        if n == 0 {
            return Err(ProbeArtifactError::InvalidShape(
                "n_embd must be > 0".into(),
            ));
        }
        if lm_head.is_empty() || !lm_head.len().is_multiple_of(n) {
            return Err(ProbeArtifactError::InvalidShape(format!(
                "lm_head length {} not divisible by n_embd {n}",
                lm_head.len()
            )));
        }
        let vocab = lm_head.len() / n;
        let mut artifact = Self {
            mlp,
            lm_head,
            vocab,
            tap_layer,
            version,
            blake3: [0u8; COMMITMENT_LEN],
        };
        let payload = artifact.payload_bytes();
        artifact.blake3 = *blake3::hash(&payload).as_bytes();
        Ok(artifact)
    }

    /// Serialize the payload — every byte the commitment covers (magic +
    /// header + weights), WITHOUT the trailing commitment.
    pub fn payload_bytes(&self) -> Vec<u8> {
        let n = self.mlp.n_embd;
        let concat_dim = 2 * n;
        // Weight f32 count: norm(2*concat) + fc1(n*concat + n) + fc2/fc3(2*(n*n + n)) + head(vocab*n)
        let f32_count = 2 * concat_dim + n * concat_dim + n + 2 * (n * n + n) + self.vocab * n;
        let mut buf = Vec::with_capacity(HEADER_LEN + 4 * f32_count);
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&(self.tap_layer as u32).to_le_bytes());
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(&(self.vocab as u32).to_le_bytes());
        for slice in [
            &self.mlp.norm_weight[..],
            &self.mlp.norm_bias[..],
            &self.mlp.fc1_weight[..],
            &self.mlp.fc1_bias[..],
            &self.mlp.fc2_weight[..],
            &self.mlp.fc2_bias[..],
            &self.mlp.fc3_weight[..],
            &self.mlp.fc3_bias[..],
            &self.lm_head[..],
        ] {
            for &v in slice.iter() {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    /// Full wire bytes: payload + trailing BLAKE3 commitment.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = self.payload_bytes();
        buf.extend_from_slice(&self.blake3);
        buf
    }

    /// Parse and VERIFY: the commitment is re-computed over the payload and
    /// compared before any weight is returned.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProbeArtifactError> {
        if bytes.len() < HEADER_LEN + COMMITMENT_LEN {
            return Err(ProbeArtifactError::Truncated {
                field: "envelope",
                expected: HEADER_LEN + COMMITMENT_LEN,
                got: bytes.len(),
            });
        }
        if bytes[..4] != MAGIC {
            let mut magic = [0u8; 4];
            magic.copy_from_slice(&bytes[..4]);
            return Err(ProbeArtifactError::BadMagic(magic));
        }
        let format_version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if format_version != FORMAT_VERSION {
            return Err(ProbeArtifactError::UnsupportedFormatVersion(format_version));
        }

        let (payload, commitment_bytes) = bytes.split_at(bytes.len() - COMMITMENT_LEN);
        let mut commitment = [0u8; COMMITMENT_LEN];
        commitment.copy_from_slice(commitment_bytes);
        let recomputed = blake3::hash(payload);
        if recomputed.as_bytes() != &commitment {
            return Err(ProbeArtifactError::CommitmentMismatch);
        }

        let version = u64::from_le_bytes(payload[8..16].try_into().unwrap());
        let tap_layer = u32::from_le_bytes(payload[16..20].try_into().unwrap()) as usize;
        let n_embd = u32::from_le_bytes(payload[20..24].try_into().unwrap()) as usize;
        let vocab = u32::from_le_bytes(payload[24..28].try_into().unwrap()) as usize;
        if n_embd == 0 {
            return Err(ProbeArtifactError::InvalidShape(
                "n_embd must be > 0".into(),
            ));
        }
        if vocab == 0 {
            return Err(ProbeArtifactError::InvalidShape("vocab must be > 0".into()));
        }

        let concat_dim = 2 * n_embd;
        let mut off = HEADER_LEN;
        let mut take =
            |field: &'static str, count: usize| -> Result<Vec<f32>, ProbeArtifactError> {
                let byte_len = count * 4;
                let end = off
                    .checked_add(byte_len)
                    .ok_or(ProbeArtifactError::Truncated {
                        field,
                        expected: byte_len,
                        got: 0,
                    })?;
                if end > payload.len() {
                    return Err(ProbeArtifactError::Truncated {
                        field,
                        expected: byte_len,
                        got: payload.len() - off,
                    });
                }
                let slice = &payload[off..end];
                off = end;
                Ok(slice
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect())
            };

        let norm_weight = take("norm_weight", concat_dim)?;
        let norm_bias = take("norm_bias", concat_dim)?;
        let fc1_weight = take("fc1_weight", n_embd * concat_dim)?;
        let fc1_bias = take("fc1_bias", n_embd)?;
        let fc2_weight = take("fc2_weight", n_embd * n_embd)?;
        let fc2_bias = take("fc2_bias", n_embd)?;
        let fc3_weight = take("fc3_weight", n_embd * n_embd)?;
        let fc3_bias = take("fc3_bias", n_embd)?;
        let lm_head = take("lm_head", vocab * n_embd)?;

        if off != payload.len() {
            return Err(ProbeArtifactError::InvalidShape(format!(
                "trailing payload bytes: consumed {off} of {}",
                payload.len()
            )));
        }

        Ok(Self {
            mlp: LatentDynamicsMLP {
                n_embd,
                norm_weight,
                norm_bias,
                fc1_weight,
                fc1_bias,
                fc2_weight,
                fc2_bias,
                fc3_weight,
                fc3_bias,
            },
            lm_head,
            vocab,
            tap_layer,
            version,
            blake3: commitment,
        })
    }

    /// Write the full wire envelope to a file (atomicity is the caller's —
    /// this is a frozen artifact swapped in once, not a hot log).
    pub fn save_to_bin(&self, path: &Path) -> Result<(), ProbeArtifactError> {
        std::fs::write(path, self.to_bytes()).map_err(|e| ProbeArtifactError::Io(e.to_string()))
    }

    /// Read + verify an artifact from a file.
    pub fn load_from_bin(path: &Path) -> Result<Self, ProbeArtifactError> {
        let bytes = std::fs::read(path).map_err(|e| ProbeArtifactError::Io(e.to_string()))?;
        Self::from_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief_drafter::LatentDynamicsMLP;

    fn fixture(vocab: usize, n_embd: usize) -> ProbeArtifact {
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let lm_head: Vec<f32> = (0..vocab * n_embd)
            .map(|i| (i as f32) * 0.05 - 1.0)
            .collect();
        ProbeArtifact::from_parts(mlp, lm_head, 0, 7).expect("fixture artifact")
    }

    #[test]
    fn from_parts_computes_blake3_over_payload() {
        let artifact = fixture(11, 8);
        let expected = *blake3::hash(&artifact.payload_bytes()).as_bytes();
        assert_eq!(
            artifact.blake3, expected,
            "commitment must be BLAKE3(payload)"
        );
        assert_eq!(artifact.vocab, 11);
        assert_eq!(artifact.mlp.n_embd, 8);
        assert_eq!(artifact.version, 7);
    }

    #[test]
    fn from_parts_rejects_bad_shapes() {
        let mlp = LatentDynamicsMLP::random_init(8);
        // Not divisible by n_embd.
        let err = ProbeArtifact::from_parts(mlp.clone(), vec![0.0; 8 * 8 + 3], 0, 1)
            .expect_err("divisibility must be enforced");
        assert!(matches!(err, ProbeArtifactError::InvalidShape(_)));
        // Empty head.
        let err = ProbeArtifact::from_parts(mlp, Vec::new(), 0, 1)
            .expect_err("empty head must be rejected");
        assert!(matches!(err, ProbeArtifactError::InvalidShape(_)));
    }

    #[test]
    fn roundtrip_bytes_preserves_forward_output() {
        let artifact = fixture(9, 12);
        let bytes = artifact.to_bytes();
        let loaded = ProbeArtifact::from_bytes(&bytes).expect("roundtrip");

        let h: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect();
        let zeros = vec![0.0f32; 12];
        let a = artifact.mlp.forward(&h, &zeros);
        let b = loaded.mlp.forward(&h, &zeros);
        assert_eq!(a, b, "weights must survive the wire bit-exactly");
        assert_eq!(loaded.lm_head, artifact.lm_head);
        assert_eq!(loaded.vocab, artifact.vocab);
        assert_eq!(loaded.tap_layer, artifact.tap_layer);
        assert_eq!(loaded.version, artifact.version);
        assert_eq!(loaded.blake3, artifact.blake3);
    }

    #[test]
    fn roundtrip_file() {
        let artifact = fixture(5, 6);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("probe_artifact_test.bin");
        artifact.save_to_bin(&path).expect("save");
        let loaded = ProbeArtifact::load_from_bin(&path).expect("load");
        assert_eq!(
            loaded.to_bytes(),
            artifact.to_bytes(),
            "file roundtrip must be byte-exact"
        );
    }

    #[test]
    fn tampered_weight_rejected() {
        let artifact = fixture(9, 12);
        let mut bytes = artifact.to_bytes();
        let mid = bytes.len() - COMMITMENT_LEN;
        bytes[mid - 1] ^= 0x01; // flip one bit inside the weights region
        let err = ProbeArtifact::from_bytes(&bytes).expect_err("tamper must fail");
        assert_eq!(err, ProbeArtifactError::CommitmentMismatch);
    }

    #[test]
    fn tampered_commitment_rejected() {
        let artifact = fixture(9, 12);
        let mut bytes = artifact.to_bytes();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x80;
        let err = ProbeArtifact::from_bytes(&bytes).expect_err("commitment flip must fail");
        assert_eq!(err, ProbeArtifactError::CommitmentMismatch);
    }

    #[test]
    fn truncated_and_bad_magic_rejected() {
        let artifact = fixture(9, 12);
        let bytes = artifact.to_bytes();

        let err = ProbeArtifact::from_bytes(&bytes[..20]).expect_err("truncation must fail");
        assert!(matches!(err, ProbeArtifactError::Truncated { .. }));

        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        let err = ProbeArtifact::from_bytes(&bad_magic).expect_err("bad magic must fail");
        assert_eq!(err, ProbeArtifactError::BadMagic(*b"XLPA"));

        let mut bad_version = bytes;
        bad_version[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = ProbeArtifact::from_bytes(&bad_version).expect_err("unknown version must fail");
        assert_eq!(err, ProbeArtifactError::UnsupportedFormatVersion(99));
    }

    #[test]
    fn trailing_payload_bytes_rejected() {
        // A valid artifact with one extra byte appended inside the payload
        // region: the commitment still verifies (we recompute it), but the
        // weights no longer tile the payload — the shape check must catch it.
        let artifact = fixture(9, 12);
        let mut bytes = artifact.to_bytes();
        let insert_at = bytes.len() - COMMITMENT_LEN;
        bytes.insert(insert_at, 0u8);
        // Recompute the commitment so ONLY the shape check can catch this.
        let payload = &bytes[..bytes.len() - COMMITMENT_LEN];
        let commitment = *blake3::hash(payload).as_bytes();
        let tail = bytes.len() - COMMITMENT_LEN..bytes.len();
        bytes[tail].copy_from_slice(&commitment);

        let err = ProbeArtifact::from_bytes(&bytes).expect_err("extra byte must fail");
        assert!(matches!(err, ProbeArtifactError::InvalidShape(_)));
    }
}
