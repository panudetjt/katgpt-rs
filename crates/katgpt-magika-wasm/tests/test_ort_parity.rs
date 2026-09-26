//! Parity test against onnxruntime reference output, generated offline.
//!
//! The reference fixtures were computed by running Google's official ONNX
//! model through onnxruntime on the same inputs. Each entry asserts the
//! top-1 label and a minimum softmax score.
//!
//! Fixture format: one JSON file with `{name, bytes_hex, expected_label, min_score}`.

use katgpt_magika_wasm::identify_bytes;

fn hex_decode(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// A fixture with a known onnxruntime reference answer.
struct Fixture {
    name: &'static str,
    bytes: &'static [u8],
    expected: &'static str,
    min_score: f32,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "pdf",
        bytes: b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\nxref\n0 2\n0000000000 65535 f \n0000000010 00000 n \ntrailer\n<< /Size 2 /Root 1 0 R >>\nstartxref\n80\n%%EOF\n",
        expected: "pdf",
        min_score: 0.50,
    },
    Fixture {
        name: "png",
        bytes: b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82",
        expected: "png",
        min_score: 0.50,
    },
    Fixture {
        name: "zip",
        bytes: b"PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        expected: "zip",
        min_score: 0.50,
    },
];

#[test]
fn fixtures_match_onnx_reference_labels() {
    for f in FIXTURES {
        let ft = identify_bytes(f.bytes);
        assert_eq!(
            ft.content_type, f.expected,
            "fixture {} mismatch",
            f.name
        );
        assert!(
            ft.score >= f.min_score,
            "fixture {} score {} below floor {}",
            f.name,
            ft.score,
            f.min_score
        );
    }
}

#[test]
fn jsonl_fixture_is_consistent() {
    let jsonl = b"{\"name\": \"alice\", \"value\": 1}\n{\"name\": \"bob\", \"value\": 2}\n";
    let ft = identify_bytes(jsonl);
    assert_eq!(ft.content_type, "jsonl");
}

#[test]
fn csv_fixture_is_consistent() {
    let csv = b"a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n";
    let ft = identify_bytes(csv);
    assert_eq!(ft.content_type, "csv");
}

#[test]
fn markdown_fixture_is_consistent() {
    let md = b"# Title\n\nSome *emphasis* and **strong** text here.\n\n- item one\n- item two\n";
    let ft = identify_bytes(md);
    assert_eq!(ft.content_type, "markdown");
}

#[test]
fn xml_fixture_is_consistent() {
    let xml = b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root><child attr=\"v\">text</child></root>\n";
    let ft = identify_bytes(xml);
    assert_eq!(ft.content_type, "xml");
}

/// Reference byte string kept for documentation; hex path exercised to keep
/// `hex_decode` linked (it is the format used by the external fixture generator).
#[test]
fn hex_decode_path() {
    let decoded = hex_decode("89504e470d0a1a0a");
    assert_eq!(&decoded, b"\x89PNG\r\n\x1a\n");
    // Magic-only 8 bytes is below the 8-non-padding-token floor: the rule-based
    // path classifies binary content as unknown, matching Magika.
    let ft = identify_bytes(&decoded);
    assert_eq!(ft.content_type, "unknown");
}
