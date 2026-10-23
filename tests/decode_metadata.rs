use xl_view::decode::{DecodeError, DecodeLimits, DecodeResult, decode_file, decode_memory};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
}

fn limits() -> DecodeLimits {
    DecodeLimits::from_memory_ceiling_mib(64)
}

fn append_box(container: &mut Vec<u8>, ty: [u8; 4], payload: &[u8], eof: bool) {
    container.extend_from_slice(
        &if eof {
            0_u32
        } else {
            u32::try_from(payload.len() + 8).unwrap()
        }
        .to_be_bytes(),
    );
    container.extend_from_slice(&ty);
    container.extend_from_slice(payload);
}

fn container(raw: &[u8]) -> Vec<u8> {
    let mut bytes = b"\0\0\0\x0cJXL \r\n\x87\n".to_vec();
    append_box(&mut bytes, *b"ftyp", b"jxl \0\0\0\0jxl ", false);
    append_box(&mut bytes, *b"jxlc", raw, false);
    bytes
}

fn xmp() -> &'static [u8] {
    br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="3"/></rdf:RDF></x:xmpmeta>"#
}

fn decode_file_bytes(bytes: &[u8], name: &str) -> DecodeResult {
    let path = std::env::temp_dir().join(format!("xl-view-{name}-{}.jxl", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    let result = decode_file(&path, limits());
    std::fs::remove_file(path).unwrap();
    result
}

#[test]
fn eof_metadata_limit_includes_buffered_bytes() {
    let base = container(&fixture("linear-bt2020.jxl"));
    let limits = DecodeLimits {
        memory_ceiling_bytes: 1024,
    };
    for ty in [*b"Exif", *b"xml "] {
        let mut bytes = base.clone();
        append_box(&mut bytes, ty, &[0; 1024], true);
        // Complete but invalid TIFF/XML remains a nonfatal parse error.
        assert!(decode_memory(&bytes, limits).is_ok());

        let mut bytes = base.clone();
        append_box(&mut bytes, ty, &[0; 1025], true);
        assert!(matches!(
            decode_memory(&bytes, limits),
            Err(DecodeError::MetadataMemoryLimit {
                required: 1025,
                limit: 1024
            })
        ));
    }
}

#[test]
fn large_sparse_eof_metadata_is_rejected_with_a_typed_limit() {
    use std::io::Write;

    let mut bytes = container(&fixture("linear-bt2020.jxl"));
    append_box(&mut bytes, *b"xml ", &[], true);
    let path = std::env::temp_dir().join(format!(
        "xl-view-sparse-metadata-{}.jxl",
        std::process::id()
    ));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&bytes).unwrap();
    file.set_len(4 * 1024 * 1024 * 1024).unwrap();
    drop(file);
    let result = decode_file(&path, DecodeLimits::from_memory_ceiling_mib(1));
    std::fs::remove_file(path).unwrap();
    assert!(
        matches!(result, Err(DecodeError::MetadataMemoryLimit { required, limit }) if required == limit + 1 && limit == 1024 * 1024)
    );
}

#[test]
fn ignored_eof_boxes_preserve_pixels_and_completed_metadata() {
    let raw = fixture("linear-bt2020.jxl");
    let expected = decode_memory(&raw, limits()).unwrap();
    for (ty, payload) in [
        (*b"junk", Vec::new()),
        (*b"jumb", vec![0; 16_384]),
        (*b"brob", b"jumbignored payload".to_vec()),
    ] {
        let mut bytes = container(&raw);
        append_box(&mut bytes, *b"xml ", xmp(), false);
        append_box(&mut bytes, ty, &payload, true);
        for decoded in [
            decode_memory(&bytes, limits()),
            decode_file_bytes(&bytes, "ignored-eof-box"),
        ] {
            let decoded = decoded.unwrap();
            assert_eq!(
                decoded.store.coarse_pixels(),
                expected.store.coarse_pixels()
            );
            assert_eq!(decoded.metadata.xmp.as_ref().unwrap().rating, Some(3.0));
        }
    }
}

#[test]
fn truncated_finite_boxes_are_not_mistaken_for_ignored_eof_boxes() {
    let base = container(&fixture("linear-bt2020.jxl"));
    let mut finite = base.clone();
    append_box(&mut finite, *b"junk", b"abc", false);
    finite.pop();
    // Even the largest finite 64-bit length must not match the EOF sentinel.
    let mut maximum = base;
    maximum.extend_from_slice(&1_u32.to_be_bytes());
    maximum.extend_from_slice(b"junk");
    maximum.extend_from_slice(&u64::MAX.to_be_bytes());
    maximum.extend_from_slice(b"abc");
    for bytes in [finite, maximum] {
        for result in [
            decode_memory(&bytes, limits()),
            decode_file_bytes(&bytes, "finite-box"),
        ] {
            assert!(matches!(result, Err(DecodeError::Decoder(_))));
        }
    }
}

#[test]
fn finite_and_eof_metadata_are_completed_for_still_images() {
    let exif = include_bytes!("fixtures/exif-common.raw").to_vec();
    // animation.jxl has a hidden frame followed by its only displayable frame.
    // It must complete trailing metadata just like a still image.
    for raw in [fixture("linear-bt2020.jxl"), fixture("animation.jxl")] {
        for eof in [false, true] {
            for (ty, payload) in [(*b"Exif", exif.as_slice()), (*b"xml ", xmp())] {
                let mut bytes = container(&raw);
                append_box(&mut bytes, ty, payload, eof);
                let decoded = decode_memory(&bytes, limits()).unwrap();
                if ty == *b"Exif" {
                    assert_eq!(
                        decoded
                            .metadata
                            .exif
                            .as_ref()
                            .unwrap()
                            .camera_make
                            .as_deref(),
                        Some("ACME")
                    );
                } else {
                    assert_eq!(decoded.metadata.xmp.as_ref().unwrap().rating, Some(3.0));
                }
            }
        }
    }
}

#[test]
fn brotli_metadata_and_eof_buffered_payload_are_completed() {
    // Brotli stream generated with `brotli -q 5` from xmp() above.
    let compressed = include_bytes!("fixtures/xmp-rating-3.br");
    for eof in [false, true] {
        let mut bytes = container(&fixture("linear-bt2020.jxl"));
        let mut payload = b"xml ".to_vec();
        payload.extend_from_slice(compressed);
        append_box(&mut bytes, *b"brob", &payload, eof);
        assert_eq!(
            decode_memory(&bytes, limits())
                .unwrap()
                .metadata
                .xmp
                .as_ref()
                .unwrap()
                .rating,
            Some(3.0)
        );
    }
}

#[test]
fn incomplete_metadata_never_publishes_pixels_or_partial_payloads() {
    let raw = fixture("linear-bt2020.jxl");
    for ty in [b"Exif", b"xml ", b"brob", b"junk"] {
        let mut bytes = container(&raw);
        append_box(&mut bytes, *ty, xmp(), false);
        bytes.pop();
        assert!(matches!(
            decode_memory(&bytes, limits()),
            Err(DecodeError::Decoder(_))
        ));
    }
    for partial_header in 1..8 {
        let mut bytes = container(&raw);
        bytes.extend_from_slice(&[0; 8][..partial_header]);
        assert!(matches!(
            decode_memory(&bytes, limits()),
            Err(DecodeError::Decoder(_))
        ));
    }
    let mut bytes = container(&raw);
    append_box(&mut bytes, *b"brob", b"xml invalid brotli", true);
    assert!(matches!(
        decode_memory(&bytes, limits()),
        Err(DecodeError::Decoder(_))
    ));
}

#[test]
fn animation_stops_at_first_frame_with_complete_earlier_metadata() {
    let raw = fixture("animation-multi-frame.jxl");
    let expected = decode_memory(&raw, limits()).unwrap();
    let first_frame_end = (1..raw.len())
        .find(|&end| decode_memory(&raw[..end], limits()).is_ok())
        .unwrap();
    assert!(first_frame_end < raw.len());
    let mut bytes = container(&raw[..first_frame_end]);
    // Metadata later than the retained frame is intentionally not processed.
    append_box(
        &mut bytes,
        *b"xml ",
        b"truncated or invalid later XML",
        false,
    );
    let decoded = decode_memory(&bytes, limits()).unwrap();
    assert!(decoded.metadata.xmp.is_none());
    assert_eq!(
        decoded.store.coarse_pixels(),
        expected.store.coarse_pixels()
    );
    // Complete metadata before the codestream is retained even if later frames
    // are missing or corrupt. A large suffix must not affect that first frame.
    let mut bytes = b"\0\0\0\x0cJXL \r\n\x87\n".to_vec();
    append_box(&mut bytes, *b"ftyp", b"jxl \0\0\0\0jxl ", false);
    append_box(&mut bytes, *b"xml ", xmp(), false);
    let mut corrupt = raw[..first_frame_end].to_vec();
    corrupt.extend_from_slice(&[0xff; 8192]);
    append_box(&mut bytes, *b"jxlc", &corrupt, false);
    let decoded = decode_memory(&bytes, limits()).unwrap();
    assert_eq!(decoded.metadata.xmp.as_ref().unwrap().rating, Some(3.0));
    assert_eq!(
        decoded.store.coarse_pixels(),
        expected.store.coarse_pixels()
    );
}
