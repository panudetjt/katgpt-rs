//! The cross-target drift gate — `riir-chain` Issues 108/109, `katgpt-rs` 685.
//!
//! Every consumer of this crate asserts the same table. These tests are the
//! local half; the remote halves live in `riir-chain` (fair roll) and
//! `riir-neuron-db` (Merkle). A green test on one side proves nothing about
//! the other, which is the whole reason the fixture is shared rather than
//! each side keeping its own.
//!
//! Several tests here assert things *about the fixture* rather than about the
//! code — that the fallback vectors still take the fallback branch, that the
//! dividing vectors still can't. A fixture silently stops covering the branch
//! it was built for the moment someone regenerates it with different labels,
//! and a coverage claim nobody checks is the same as no coverage.

use katgpt_device_verify::fair_roll::{FairRollVerifier, combine_seed};
use katgpt_device_verify::merkle_verify::{
    EMPTY_HASH, MAX_PROOF_SIZE, compute_root_from_proof, hash_pair, roots_match, verify_proof,
    verify_proof_bounded,
};
use katgpt_device_verify::vectors::{
    FAIR_ROLL_DIVIDING_VECTORS, FAIR_ROLL_DOUBLE_REJECT_VECTORS, FAIR_ROLL_FALLBACK_VECTORS,
    FAIR_ROLL_NONDIVIDING_VECTORS, FairRollVector, MERKLE_VECTORS,
};

/// The rejection threshold, spelled the way the implementation spells it.
fn threshold(sides: u8) -> u16 {
    256u16 - (256 % u16::from(sides))
}

fn all_fair_roll() -> impl Iterator<Item = &'static FairRollVector> {
    FAIR_ROLL_DIVIDING_VECTORS
        .iter()
        .chain(FAIR_ROLL_NONDIVIDING_VECTORS)
        .chain(FAIR_ROLL_FALLBACK_VECTORS)
        .chain(FAIR_ROLL_DOUBLE_REJECT_VECTORS)
}

/// The face `roll_die` (v2) must deal: walk the hash bytes, take the first
/// one below the threshold, reduce it. Spelled locally — mirrors of the
/// rule are what turn a table+seed mismatch into a red test.
fn first_accepted_face(seed: &[u8; 32], sides: u8) -> u8 {
    let t = threshold(sides);
    let mut hash = blake3::hash(seed);
    loop {
        for &b in hash.as_bytes() {
            if u16::from(b) < t {
                return b % sides + 1;
            }
        }
        hash = blake3::hash(hash.as_bytes());
    }
}

// ── The gate itself ────────────────────────────────────────────────────

#[test]
fn fair_roll_vectors_match() {
    let mut n = 0;
    for v in all_fair_roll() {
        let got = FairRollVerifier::from_combined_seed(v.seed).roll_die(v.sides);
        assert_eq!(
            got, v.die,
            "DRIFT on {} (sides={}): pinned {} but computed {}",
            v.label, v.sides, v.die, got
        );
        n += 1;
    }
    assert_eq!(n, 35, "vector count changed — was the fixture regenerated?");
}

#[test]
fn merkle_vectors_match() {
    assert_eq!(MERKLE_VECTORS.len(), 19, "vector count changed");
    for v in MERKLE_VECTORS {
        assert!(
            verify_proof(&v.leaf, v.index, v.siblings, &v.root),
            "DRIFT on merkle vector {}",
            v.label
        );
        assert_eq!(
            compute_root_from_proof(&v.leaf, v.index, v.siblings),
            v.root,
            "root mismatch on {}",
            v.label
        );
    }
}

// ── Fixture-coverage assertions (is the table still testing what it claims?) ──

#[test]
fn dividing_vectors_cannot_reach_the_fallback() {
    for v in FAIR_ROLL_DIVIDING_VECTORS {
        assert_eq!(
            256 % u16::from(v.sides),
            0,
            "{} is in the DIVIDING set but {} does not divide 256",
            v.label,
            v.sides
        );
        assert_eq!(
            threshold(v.sides),
            256,
            "{}: threshold must be 256 so the first byte is always accepted",
            v.label
        );
    }
}

#[test]
fn nondividing_vectors_have_a_live_rejection_branch() {
    for v in FAIR_ROLL_NONDIVIDING_VECTORS {
        assert_ne!(
            256 % u16::from(v.sides),
            0,
            "{} is in the NON-DIVIDING set but {} divides 256 — the rejection \
             branch it exists to cover is dead",
            v.label,
            v.sides
        );
    }
}

#[test]
fn fallback_vectors_actually_take_the_fallback_branch() {
    for v in FAIR_ROLL_FALLBACK_VECTORS {
        let first = u16::from(blake3::hash(&v.seed).as_bytes()[0]);
        assert!(
            first >= threshold(v.sides),
            "{}: first byte {} < threshold {} — this vector no longer \
             exercises the rejection branch, so the branch is UNCOVERED",
            v.label,
            first,
            threshold(v.sides)
        );
        // v2: the first byte rejects by construction, so the decider is a
        // later byte — and it must produce the pinned face.
        assert_eq!(
            v.die,
            first_accepted_face(&v.seed, v.sides),
            "{}: pinned die is not the first accepted byte's face",
            v.label
        );
    }
}

#[test]
fn double_reject_vectors_take_the_v2_retry_path() {
    for v in FAIR_ROLL_DOUBLE_REJECT_VECTORS {
        let hash = blake3::hash(&v.seed);
        let t = threshold(v.sides);
        assert!(
            u16::from(hash.as_bytes()[0]) >= t,
            "{}: hash[0] accepted — not a double-rejection seed",
            v.label
        );
        assert!(
            u16::from(hash.as_bytes()[1]) >= t,
            "{}: hash[1] accepted — the retry path is not exercised",
            v.label
        );
        // The decider is at index >= 2 (or the chained hash) and must match
        // the pinned face.
        assert_eq!(
            v.die,
            first_accepted_face(&v.seed, v.sides),
            "{}: pinned die is not the first accepted byte's face",
            v.label
        );
        // And it must DISAGREE with v1's unconditional fallback byte — these
        // vectors were searched for that property, so a regression to v1
        // flips every row here rather than passing silently.
        let v1_face = hash.as_bytes()[1] % v.sides + 1;
        assert_ne!(
            v.die, v1_face,
            "{}: v1 and v2 agree — this vector no longer discriminates the seams",
            v.label
        );
    }
}

// ── Invariants ─────────────────────────────────────────────────────────

#[test]
fn roll_die_is_always_in_range() {
    for v in all_fair_roll() {
        let d = FairRollVerifier::from_combined_seed(v.seed).roll_die(v.sides);
        assert!(
            (1..=v.sides).contains(&d),
            "{}: die {} outside 1..={}",
            v.label,
            d,
            v.sides
        );
    }
}

#[test]
fn zero_sides_is_refused_not_a_reset() {
    let vfy = FairRollVerifier::from_combined_seed([7u8; 32]);
    assert_eq!(vfy.checked_roll_die(0), None);
    assert!(
        !vfy.verify_die(0, 1),
        "a malformed sides must fail verification"
    );
}

#[test]
fn verify_die_accepts_only_the_dealt_value() {
    for v in all_fair_roll() {
        let vfy = FairRollVerifier::from_combined_seed(v.seed);
        assert!(vfy.verify_die(v.sides, v.die), "{}", v.label);
        // Every other face on the die must be refused.
        for face in 1..=v.sides {
            if face != v.die {
                assert!(
                    !vfy.verify_die(v.sides, face),
                    "{}: accepted a face it did not deal ({face})",
                    v.label
                );
            }
        }
    }
}

#[test]
fn combine_seed_is_blake3_of_the_concatenation() {
    let alpha = [0xAAu8; 32];
    let beta = [0xBBu8; 32];
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(&alpha);
    buf[32..].copy_from_slice(&beta);
    assert_eq!(combine_seed(&alpha, &beta), *blake3::hash(&buf).as_bytes());
    // Order matters — a device that swaps the halves gets a different seed,
    // which is what stops a node from re-labelling the reveal.
    assert_ne!(combine_seed(&alpha, &beta), combine_seed(&beta, &alpha));
}

#[test]
fn roll_unit_is_in_the_half_open_unit_interval() {
    for v in all_fair_roll() {
        let u = FairRollVerifier::from_combined_seed(v.seed).roll_unit();
        assert!(
            (0.0..1.0).contains(&u),
            "{}: roll_unit {u} out of range",
            v.label
        );
    }
}

// ── Merkle ─────────────────────────────────────────────────────────────

#[test]
fn empty_hash_is_the_pinned_sentinel_and_not_blake3_of_nothing() {
    // The bytes riir-neuron-db actually pairs orphans with. Transcribed
    // independently from `riir-neuron-db/src/merkle.rs:42` so that a change
    // on either side reds this test rather than silently forking roots.
    const UPSTREAM: [u8; 32] = [
        0xAF, 0xE4, 0x52, 0xD4, 0x88, 0x1B, 0x85, 0x0D, 0x41, 0x9F, 0x2E, 0x3A, 0x3C, 0x73, 0xE2,
        0x2F, 0x30, 0x6B, 0x5C, 0xE9, 0x5E, 0x6B, 0xB0, 0x93, 0x5A, 0x36, 0x75, 0xF3, 0x01, 0x31,
        0x1B, 0x09,
    ];
    assert_eq!(
        EMPTY_HASH, UPSTREAM,
        "EMPTY_HASH drifted from riir-neuron-db's orphan-pairing sentinel"
    );

    // Regression guard, not a curiosity. Upstream's comment calls this
    // "BLAKE3 of the empty input"; it is not, and a well-meaning cleanup that
    // "fixes" the constant to match its own comment would fork the root of
    // every odd-sized tree ever committed. If this assertion ever flips,
    // someone did exactly that.
    assert_ne!(
        EMPTY_HASH,
        *blake3::hash(b"").as_bytes(),
        "someone 'corrected' EMPTY_HASH to BLAKE3(\"\") — that forks every \
         odd-tree root in history; the COMMENT is what is wrong, not the bytes"
    );
}

#[test]
fn hash_pair_is_order_sensitive() {
    let a = [1u8; 32];
    let b = [2u8; 32];
    assert_ne!(
        hash_pair(&a, &b),
        hash_pair(&b, &a),
        "an order-insensitive pair hash lets a proof be replayed with the \
         siblings mirrored"
    );
}

#[test]
fn a_tampered_leaf_fails_verification() {
    for v in MERKLE_VECTORS {
        let mut bad = v.leaf;
        bad[0] ^= 0x01;
        assert!(
            !verify_proof(&bad, v.index, v.siblings, &v.root),
            "{}: a flipped leaf bit still verified",
            v.label
        );
    }
}

#[test]
fn a_tampered_sibling_fails_verification() {
    for v in MERKLE_VECTORS.iter().filter(|v| !v.siblings.is_empty()) {
        let mut sibs = v.siblings.to_vec();
        sibs[0][0] ^= 0x01;
        assert!(
            !verify_proof(&v.leaf, v.index, &sibs, &v.root),
            "{}: a flipped sibling bit still verified",
            v.label
        );
    }
}

#[test]
fn a_wrong_index_fails_verification() {
    // Only meaningful where flipping the low bit changes the hash order —
    // i.e. a proof with at least one sibling.
    for v in MERKLE_VECTORS.iter().filter(|v| !v.siblings.is_empty()) {
        let wrong = v.index ^ 1;
        assert!(
            !verify_proof(&v.leaf, wrong, v.siblings, &v.root),
            "{}: index {} verified against a proof for {}",
            v.label,
            wrong,
            v.index
        );
    }
}

#[test]
fn an_over_deep_proof_is_refused_rather_than_walked() {
    let leaf = [3u8; 32];
    let sibs = vec![[4u8; 32]; MAX_PROOF_SIZE + 1];
    let root = compute_root_from_proof(&leaf, 0, &sibs);
    // The unbounded verifier honours it; the bounded one refuses the work.
    assert!(verify_proof(&leaf, 0, &sibs, &root));
    assert!(
        !verify_proof_bounded(&leaf, 0, &sibs, &root),
        "an attacker-supplied path longer than MAX_PROOF_SIZE must be \
         refused, not walked — unbounded work in a Glacial budget"
    );
    // A proof exactly at the cap is still legitimate.
    let ok = vec![[4u8; 32]; MAX_PROOF_SIZE];
    let ok_root = compute_root_from_proof(&leaf, 0, &ok);
    assert!(verify_proof_bounded(&leaf, 0, &ok, &ok_root));
}

#[test]
fn roots_match_is_plain_equality() {
    let a = [9u8; 32];
    let mut b = a;
    assert!(roots_match(&a, &b));
    b[31] ^= 0x80;
    assert!(!roots_match(&a, &b));
}
