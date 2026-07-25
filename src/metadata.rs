//! Normalized source metadata which is independent of the image decoder.

use std::fmt::Write as _;

use exif::{Error as ExifError, Exif, In, Tag, Value};

const MAX_EXIF_TEXT_CHARACTERS: usize = 256;
const MAX_EXIF_BYTES: usize = 16 * 1024 * 1024;
const MAX_XMP_BYTES: usize = 4 * 1024 * 1024;
const MAX_XMP_NODES: u32 = 4_096;
const XMP_BASIC_NAMESPACE: &str = "http://ns.adobe.com/xap/1.0/";

/// Common fields parsed from an EXIF payload capped at 16 MiB.
#[derive(Clone, Debug, PartialEq)]
pub struct ExifMetadata {
    pub aperture_f_number: Option<f64>,
    pub artist: Option<String>,
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub captured_at: Option<String>,
    pub copyright: Option<String>,
    pub exposure_bias_ev: Option<f64>,
    pub exposure_time_seconds: Option<f64>,
    pub focal_length_mm: Option<f64>,
    pub iso_speed: Option<u32>,
    pub lens_make: Option<String>,
    pub lens_model: Option<String>,
    pub parse_error: Option<String>,
    pub software: Option<String>,
}

/// XMP fields used by the application.
///
/// The presence of this value records that the JPEG XL container had an XML
/// box even when it contained no supported properties or could not be parsed.
#[derive(Clone, Debug, PartialEq)]
pub struct XmpMetadata {
    pub parse_error: Option<String>,
    pub rating: Option<f64>,
}

impl ExifMetadata {
    fn empty() -> Self {
        Self {
            aperture_f_number: None,
            artist: None,
            camera_make: None,
            camera_model: None,
            captured_at: None,
            copyright: None,
            exposure_bias_ev: None,
            exposure_time_seconds: None,
            focal_length_mm: None,
            iso_speed: None,
            lens_make: None,
            lens_model: None,
            parse_error: None,
            software: None,
        }
    }
}

/// Parses common TIFF/EXIF fields without retaining the source payload.
///
/// Invalid EXIF is returned with `parse_error` set instead of failing image
/// decoding. Valid fields from partially malformed EXIF are retained alongside
/// a description of the ignored errors.
#[must_use]
pub fn parse_exif(raw_payload: &[u8], tiff_header_offset: u32) -> ExifMetadata {
    let mut metadata = ExifMetadata::empty();
    if raw_payload.len() > MAX_EXIF_BYTES {
        metadata.parse_error = Some(format!(
            "EXIF payload is {} bytes; limit is {MAX_EXIF_BYTES} bytes",
            raw_payload.len()
        ));
        return metadata;
    }
    if let Err(error) = parse_exif_fields(&mut metadata, raw_payload, tiff_header_offset) {
        metadata.parse_error = Some(error);
    }
    metadata
}

fn parse_exif_fields(
    metadata: &mut ExifMetadata,
    raw_payload: &[u8],
    tiff_header_offset: u32,
) -> Result<(), String> {
    let header_offset = usize::try_from(tiff_header_offset)
        .map_err(|_| "TIFF header offset is too large".to_owned())?;
    let tiff = raw_payload
        .get(header_offset..)
        .ok_or_else(|| "TIFF header offset is outside the EXIF payload".to_owned())?;

    let mut reader = exif::Reader::new();
    reader.continue_on_error(true);
    let exif = match reader.read_raw(tiff.to_vec()) {
        Ok(exif) => exif,
        Err(ExifError::PartialResult(partial)) => {
            let (exif, errors) = partial.into_inner();
            metadata.parse_error = Some(format_partial_errors(&errors));
            exif
        }
        Err(error) => return Err(error.to_string()),
    };

    metadata.camera_make = ascii(&exif, Tag::Make);
    metadata.camera_model = ascii(&exif, Tag::Model);
    metadata.software = ascii(&exif, Tag::Software);
    metadata.artist = ascii(&exif, Tag::Artist);
    metadata.copyright = ascii(&exif, Tag::Copyright);
    metadata.captured_at = ascii(&exif, Tag::DateTimeOriginal)
        .or_else(|| ascii(&exif, Tag::DateTimeDigitized))
        .or_else(|| ascii(&exif, Tag::DateTime));
    metadata.exposure_time_seconds = unsigned_rational(&exif, Tag::ExposureTime);
    metadata.aperture_f_number = unsigned_rational(&exif, Tag::FNumber);
    metadata.iso_speed =
        unsigned(&exif, Tag::PhotographicSensitivity).or_else(|| unsigned(&exif, Tag::ISOSpeed));
    metadata.exposure_bias_ev = signed_rational(&exif, Tag::ExposureBiasValue);
    metadata.focal_length_mm = unsigned_rational(&exif, Tag::FocalLength);
    metadata.lens_make = ascii(&exif, Tag::LensMake);
    metadata.lens_model = ascii(&exif, Tag::LensModel);
    Ok(())
}

fn ascii(exif: &Exif, tag: Tag) -> Option<String> {
    let Value::Ascii(values) = &exif.get_field(tag, In::PRIMARY)?.value else {
        return None;
    };
    let mut text = String::new();
    for value in values {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&String::from_utf8_lossy(value));
    }
    let text = text
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    let normalized = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(MAX_EXIF_TEXT_CHARACTERS)
        .collect::<String>();
    (!normalized.is_empty()).then_some(normalized)
}

fn unsigned(exif: &Exif, tag: Tag) -> Option<u32> {
    exif.get_field(tag, In::PRIMARY)?.value.get_uint(0)
}

fn unsigned_rational(exif: &Exif, tag: Tag) -> Option<f64> {
    let Value::Rational(values) = &exif.get_field(tag, In::PRIMARY)?.value else {
        return None;
    };
    let value = values.first()?;
    (value.denom != 0).then(|| value.to_f64())
}

fn signed_rational(exif: &Exif, tag: Tag) -> Option<f64> {
    let Value::SRational(values) = &exif.get_field(tag, In::PRIMARY)?.value else {
        return None;
    };
    let value = values.first()?;
    (value.denom != 0).then(|| value.to_f64())
}

fn format_partial_errors(errors: &[ExifError]) -> String {
    let mut details = errors
        .iter()
        .take(3)
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    if errors.len() > 3 {
        write!(details, "; and {} more", errors.len() - 3).unwrap();
    }
    if details.is_empty() {
        "partially parsed EXIF".to_owned()
    } else {
        format!("partially parsed EXIF: {details}")
    }
}

/// Parses supported fields from a JPEG XL XML metadata box.
///
/// XMP is treated as untrusted input: DTDs are rejected and the XML tree has a
/// fixed node limit. Invalid XML or property values are retained as a
/// diagnostic instead of failing image decoding.
#[must_use]
pub fn parse_xmp(raw_payload: &[u8]) -> XmpMetadata {
    match parse_xmp_rating(raw_payload) {
        Ok(rating) => XmpMetadata {
            parse_error: None,
            rating,
        },
        Err(error) => XmpMetadata {
            parse_error: Some(error),
            rating: None,
        },
    }
}

fn parse_xmp_rating(raw_payload: &[u8]) -> Result<Option<f64>, String> {
    if raw_payload.len() > MAX_XMP_BYTES {
        return Err(format!(
            "XMP payload is {} bytes; limit is {MAX_XMP_BYTES} bytes",
            raw_payload.len()
        ));
    }
    let xml = std::str::from_utf8(raw_payload)
        .map_err(|error| format!("XMP is not valid UTF-8: {error}"))?;
    let options = roxmltree::ParsingOptions {
        nodes_limit: MAX_XMP_NODES,
        ..roxmltree::ParsingOptions::default()
    };
    let document = roxmltree::Document::parse_with_options(xml, options)
        .map_err(|error| format!("invalid XMP XML: {error}"))?;

    for node in document.descendants().filter(roxmltree::Node::is_element) {
        if let Some(value) = node.attribute((XMP_BASIC_NAMESPACE, "Rating")) {
            return parse_rating_value(value).map(Some);
        }
        if node.has_tag_name((XMP_BASIC_NAMESPACE, "Rating")) {
            let value = node
                .text()
                .ok_or_else(|| "XMP rating has no value".to_owned())?;
            return parse_rating_value(value).map(Some);
        }
    }
    Ok(None)
}

fn parse_rating_value(value: &str) -> Result<f64, String> {
    let rating = value
        .trim()
        .parse::<f64>()
        .map_err(|_| "XMP rating is not numeric".to_owned())?;
    if rating.to_bits() == (-1.0_f64).to_bits() || (0.0..=5.0).contains(&rating) {
        Ok(rating)
    } else {
        Err(format!(
            "XMP rating {rating} is outside the supported -1 or 0..=5 range"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_little_endian_common_fields() {
        let payload = test_payload();
        let exif = parse_exif(&payload, 0);
        assert_eq!(exif.camera_make.as_deref(), Some("ACME"));
        assert_eq!(exif.camera_model.as_deref(), Some("Photon 1"));
        assert_eq!(exif.lens_model.as_deref(), Some("Prime 50"));
        assert_eq!(exif.captured_at.as_deref(), Some("2026:07:13 12:34:56"));
        assert_eq!(exif.iso_speed, Some(200));
        assert_eq!(exif.exposure_time_seconds, Some(1.0 / 125.0));
        assert_eq!(exif.aperture_f_number, Some(2.8));
        assert_eq!(exif.focal_length_mm, Some(50.0));
        assert_eq!(exif.exposure_bias_ev, Some(-1.0 / 3.0));
        assert_eq!(exif.parse_error, None);
    }

    #[test]
    fn honors_tiff_header_offset() {
        let mut payload = b"JXL prefix".to_vec();
        let offset = u32::try_from(payload.len()).unwrap();
        payload.extend_from_slice(&test_payload());
        let exif = parse_exif(&payload, offset);
        assert_eq!(exif.camera_make.as_deref(), Some("ACME"));
        assert_eq!(exif.parse_error, None);
    }

    #[test]
    fn malformed_exif_is_non_fatal() {
        let exif = parse_exif(b"not TIFF", 0);
        assert!(exif.parse_error.is_some());
    }

    #[test]
    fn oversized_exif_is_not_parsed() {
        let payload = vec![0; MAX_EXIF_BYTES + 1];
        let exif = parse_exif(&payload, 0);
        assert!(exif.parse_error.is_some());
    }

    #[test]
    fn parses_big_endian_inline_ascii() {
        let payload = [
            b'M', b'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x0f, 0, 2, 0, 0, 0, 4, b'A', b'C', 0, 0, 0,
            0, 0, 0,
        ];
        let exif = parse_exif(&payload, 0);
        assert_eq!(exif.camera_make.as_deref(), Some("AC"));
        assert_eq!(exif.parse_error, None);
    }

    #[test]
    fn parses_xmp_rating_attribute_by_namespace_instead_of_prefix() {
        let xmp = parse_xmp(
            br#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                <rdf:Description xmlns:metadata="http://ns.adobe.com/xap/1.0/"
                    metadata:Rating="4"/>
            </rdf:RDF>"#,
        );
        assert_eq!(xmp.rating, Some(4.0));
        assert_eq!(xmp.parse_error, None);
    }

    #[test]
    fn parses_xmp_rating_element_and_fractional_value() {
        let xmp = parse_xmp(
            br#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                    xmlns:xmp="http://ns.adobe.com/xap/1.0/">
                <rdf:Description><xmp:Rating>4.5</xmp:Rating></rdf:Description>
            </rdf:RDF>"#,
        );
        assert_eq!(xmp.rating, Some(4.5));
        assert_eq!(xmp.parse_error, None);
    }

    #[test]
    fn ignores_rating_from_an_unrelated_xml_namespace() {
        let xmp = parse_xmp(
            br#"<other:Rating xmlns:other="https://example.invalid/not-xmp">5</other:Rating>"#,
        );
        assert_eq!(xmp.rating, None);
        assert_eq!(xmp.parse_error, None);
    }

    #[test]
    fn invalid_xmp_is_non_fatal() {
        let malformed = parse_xmp(b"<xmp:Rating>");
        assert_eq!(malformed.rating, None);
        assert!(malformed.parse_error.is_some());

        let dtd = parse_xmp(b"<!DOCTYPE xmp><xmp/>");
        assert_eq!(dtd.rating, None);
        assert!(dtd.parse_error.is_some());

        let out_of_range =
            parse_xmp(br#"<xmp:Rating xmlns:xmp="http://ns.adobe.com/xap/1.0/">6</xmp:Rating>"#);
        assert_eq!(out_of_range.rating, None);
        assert!(out_of_range.parse_error.is_some());

        let oversized_payload = vec![b' '; MAX_XMP_BYTES + 1];
        let oversized = parse_xmp(&oversized_payload);
        assert_eq!(oversized.rating, None);
        assert!(oversized.parse_error.is_some());
    }

    fn test_payload() -> Vec<u8> {
        let mut data = vec![b'I', b'I', 42, 0, 8, 0, 0, 0];
        let ifd0 = data.len();
        data.extend_from_slice(&3_u16.to_le_bytes());
        data.resize(data.len() + 3 * 12 + 4, 0);
        let make_offset = append(&mut data, b"ACME\0");
        let model_offset = append(&mut data, b"Photon 1\0");
        let exif_ifd_offset = u32::try_from(data.len()).unwrap();
        write_entry(&mut data, ifd0 + 2, 0x010f, 2, 5, make_offset);
        write_entry(&mut data, ifd0 + 14, 0x0110, 2, 9, model_offset);
        write_entry(&mut data, ifd0 + 26, 0x8769, 4, 1, exif_ifd_offset);

        let exif_ifd = data.len();
        data.extend_from_slice(&7_u16.to_le_bytes());
        data.resize(data.len() + 7 * 12 + 4, 0);
        let exposure = append_rational(&mut data, 1, 125);
        let aperture = append_rational(&mut data, 28, 10);
        let captured = append(&mut data, b"2026:07:13 12:34:56\0");
        let bias = append_signed_rational(&mut data, -1, 3);
        let focal = append_rational(&mut data, 50, 1);
        let lens = append(&mut data, b"Prime 50\0");
        write_entry(&mut data, exif_ifd + 2, 0x829a, 5, 1, exposure);
        write_entry(&mut data, exif_ifd + 14, 0x829d, 5, 1, aperture);
        write_entry(&mut data, exif_ifd + 26, 0x8827, 3, 1, 200);
        write_entry(&mut data, exif_ifd + 38, 0x9003, 2, 20, captured);
        write_entry(&mut data, exif_ifd + 50, 0x9204, 10, 1, bias);
        write_entry(&mut data, exif_ifd + 62, 0x920a, 5, 1, focal);
        write_entry(&mut data, exif_ifd + 74, 0xa434, 2, 9, lens);
        data
    }

    fn append(data: &mut Vec<u8>, value: &[u8]) -> u32 {
        let offset = u32::try_from(data.len()).unwrap();
        data.extend_from_slice(value);
        offset
    }

    fn append_rational(data: &mut Vec<u8>, numerator: u32, denominator: u32) -> u32 {
        let offset = u32::try_from(data.len()).unwrap();
        data.extend_from_slice(&numerator.to_le_bytes());
        data.extend_from_slice(&denominator.to_le_bytes());
        offset
    }

    fn append_signed_rational(data: &mut Vec<u8>, numerator: i32, denominator: i32) -> u32 {
        let offset = u32::try_from(data.len()).unwrap();
        data.extend_from_slice(&numerator.to_le_bytes());
        data.extend_from_slice(&denominator.to_le_bytes());
        offset
    }

    fn write_entry(
        data: &mut [u8],
        offset: usize,
        tag: u16,
        field_type: u16,
        count: u32,
        value: u32,
    ) {
        data[offset..offset + 2].copy_from_slice(&tag.to_le_bytes());
        data[offset + 2..offset + 4].copy_from_slice(&field_type.to_le_bytes());
        data[offset + 4..offset + 8].copy_from_slice(&count.to_le_bytes());
        data[offset + 8..offset + 12].copy_from_slice(&value.to_le_bytes());
    }
}
