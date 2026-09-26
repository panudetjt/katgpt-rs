//! Parity tests verifying that katgpt-magika-wasm detects real files identically
//! to Google Magika's reference model.

use katgpt_magika_wasm::{identify_bytes, postprocess::type_info_of};

#[test]
fn test_pdf_detection() {
    let pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\nxref\n0 2\n0000000000 65535 f \n0000000010 00000 n \ntrailer\n<< /Size 2 /Root 1 0 R >>\nstartxref\n80\n%%EOF\n";
    let ft = identify_bytes(pdf);
    assert_eq!(ft.content_type, "pdf");
    assert!(ft.score >= 0.50, "score was {}", ft.score);
    let info = type_info_of(&ft);
    assert_eq!(info.mime_type, "application/pdf");
    assert_eq!(info.group, "document");
}

#[test]
fn test_png_detection() {
    let png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82";
    let ft = identify_bytes(png);
    assert_eq!(ft.content_type, "png");
    assert!(ft.score >= 0.50, "score was {}", ft.score);
    let info = type_info_of(&ft);
    assert_eq!(info.mime_type, "image/png");
    assert_eq!(info.group, "image");
}

#[test]
fn test_python_detection() {
    let py = b"import os\nimport sys\n\ndef main():\n    print('hello world')\n\nif __name__ == '__main__':\n    main()\n";
    let ft = identify_bytes(py);
    assert_eq!(ft.content_type, "python");
    assert!(ft.score >= 0.50, "score was {}", ft.score);
    let info = type_info_of(&ft);
    assert_eq!(info.mime_type, "text/x-python");
    assert_eq!(info.group, "code");
    assert!(info.is_text);
}

#[test]
fn test_rust_detection() {
    let rs = b"fn main() {\n    println!(\"hello world\");\n}\n";
    let ft = identify_bytes(rs);
    assert_eq!(ft.content_type, "rust");
    assert!(ft.score >= 0.50, "score was {}", ft.score);
    let info = type_info_of(&ft);
    assert_eq!(info.mime_type, "application/x-rust");
    assert_eq!(info.group, "code");
    assert!(info.is_text);
}

#[test]
fn test_html_detection() {
    let html = b"<!DOCTYPE html>\n<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>\n";
    let ft = identify_bytes(html);
    assert_eq!(ft.content_type, "html");
    assert!(ft.score >= 0.50, "score was {}", ft.score);
    let info = type_info_of(&ft);
    assert_eq!(info.mime_type, "text/html");
    assert_eq!(info.group, "code");
    assert!(info.is_text);
}

#[test]
fn test_empty_and_short_fallbacks() {
    let empty = identify_bytes(b"");
    assert_eq!(empty.content_type, "empty");
    assert_eq!(empty.score, 1.0);
    assert_eq!(type_info_of(&empty).mime_type, "inode/x-empty");

    let short_txt = identify_bytes(b"a");
    assert_eq!(short_txt.content_type, "txt");
    assert_eq!(short_txt.score, 1.0);
    assert_eq!(type_info_of(&short_txt).mime_type, "text/plain");

    let short_bin = identify_bytes(&[0, 255, 128, 200]);
    assert_eq!(short_bin.content_type, "unknown");
    assert_eq!(short_bin.score, 1.0);
    assert_eq!(type_info_of(&short_bin).mime_type, "application/octet-stream");
}
