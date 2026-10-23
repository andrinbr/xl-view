use std::path::Path;
use std::time::Duration;

use xl_view::color::SourceIntensityTarget;
use xl_view::decode::{DecodedImage, SourceDynamicRange};
use xl_view::metadata::{ExifMetadata, XmpMetadata};

use super::files::file_name;
use crate::gpu::OverlaySection;
use xl_view::units::{format_mib, milliseconds, u64_from_usize};

#[derive(Debug)]
pub(super) struct LoadedImageSummary {
    pub(super) decode_timing: DecodeTiming,
    pub(super) dimensions: (u32, u32),
    pub(super) exif: Option<ExifMetadata>,
    pub(super) file_name: String,
    pub(super) hdr_source: bool,
    pub(super) memory_bytes: usize,
    pub(super) source_color_space: String,
    pub(super) source_peak: SourceIntensityTarget,
    pub(super) peak_reference: Option<String>,
    pub(super) source_transfer: String,
    pub(super) xmp: Option<XmpMetadata>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DecodeTiming {
    Measured(Duration),
    CacheHit(Duration),
}

impl LoadedImageSummary {
    /// Builds the metadata overlay, with `display_rows` describing the output.
    pub(super) fn overlay_sections(
        &self,
        display_rows: Vec<(String, String)>,
    ) -> Vec<OverlaySection> {
        let mut sections = vec![
            OverlaySection {
                title: "IMAGE",
                rows: vec![
                    ("File".to_owned(), self.file_name.clone()),
                    (
                        "Dimensions".to_owned(),
                        dimensions_summary(self.dimensions.0, self.dimensions.1),
                    ),
                    (
                        "Decode".to_owned(),
                        decode_summary(self.decode_timing, self.memory_bytes),
                    ),
                ],
            },
            OverlaySection {
                title: "SOURCE",
                rows: vec![
                    (
                        "Range".to_owned(),
                        source_range_summary(self.hdr_source, &self.source_transfer),
                    ),
                    ("Color space".to_owned(), self.source_color_space.clone()),
                    ("Source peak".to_owned(), self.source_peak_summary()),
                ],
            },
            OverlaySection {
                title: "DISPLAY & VIEWER",
                rows: display_rows,
            },
        ];
        let capture_rows = capture_metadata_rows(self.exif.as_ref());
        if !capture_rows.is_empty() {
            sections.push(OverlaySection {
                title: "CAPTURE",
                rows: capture_rows,
            });
        }
        let attribution_rows = attribution_metadata_rows(self.exif.as_ref(), self.xmp.as_ref());
        if !attribution_rows.is_empty() {
            sections.push(OverlaySection {
                title: "ATTRIBUTION",
                rows: attribution_rows,
            });
        }
        sections
    }

    fn source_peak_summary(&self) -> String {
        let peak = self.source_peak.nits();
        match &self.peak_reference {
            Some(reference) => format!("{peak:.0} nits ({reference})"),
            None => format!("{peak:.0} nits"),
        }
    }

    pub(super) fn from_decoded(
        path: &Path,
        image: &DecodedImage,
        decode_timing: DecodeTiming,
    ) -> Self {
        let (source_color_space, source_transfer) =
            source_encoding_details(&image.metadata.color_encoding);
        Self {
            decode_timing,
            dimensions: (image.width, image.height),
            file_name: file_name(path),
            exif: image.metadata.exif.clone(),
            hdr_source: image.hdr_source,
            memory_bytes: image.memory_cost_bytes,
            source_color_space,
            source_transfer,
            source_peak: image.source_peak,
            peak_reference: match image.source_dynamic_range {
                SourceDynamicRange::Sdr => None,
                SourceDynamicRange::Linear => image.linear_unit.map(|unit| {
                    let suffix = if unit.assumed { ", assumed" } else { "" };
                    format!("unit {:.0} nits{suffix}", unit.nits)
                }),
                SourceDynamicRange::Pq | SourceDynamicRange::Hlg => {
                    Some(format!("target {:.0} nits", image.intensity_target.nits()))
                }
            },
            xmp: image.metadata.xmp.clone(),
        }
    }
}

fn source_encoding_details(encoding: &xl_view::decode::SourceColorEncoding) -> (String, String) {
    match encoding {
        xl_view::decode::SourceColorEncoding::Enumerated {
            color_space,
            white_point,
            primaries,
            transfer_function,
        } => {
            let primaries = primaries
                .as_ref()
                .map_or_else(String::new, |primaries| format!(", {primaries} primaries"));
            (
                format!("{color_space}{primaries}, {white_point} white"),
                transfer_function.clone(),
            )
        }
        xl_view::decode::SourceColorEncoding::Icc {
            color_space,
            profile_bytes,
        } => (
            format!("ICC profile, {color_space} ({profile_bytes} bytes)"),
            "ICC-defined".to_owned(),
        ),
    }
}

fn source_range_summary(hdr: bool, transfer_function: &str) -> String {
    let range = if hdr { "HDR" } else { "SDR" };
    format!("{range} ({transfer_function})")
}

fn decode_summary(decode_timing: DecodeTiming, decoded_bytes: usize) -> String {
    let timing = match decode_timing {
        DecodeTiming::Measured(duration) => {
            format!("{:.1} ms", milliseconds(duration))
        }
        DecodeTiming::CacheHit(duration) => {
            format!("{:.1} ms (cached)", milliseconds(duration))
        }
    };
    format!(
        "{timing}, {} MiB",
        format_mib(u64_from_usize(decoded_bytes)),
    )
}

fn dimensions_summary(width: u32, height: u32) -> String {
    let megapixels = f64::from(width) * f64::from(height) / 1_000_000.0;
    format!("{width} x {height} ({megapixels:.1} MP)")
}

fn capture_metadata_rows(exif: Option<&ExifMetadata>) -> Vec<(String, String)> {
    let Some(exif) = exif else {
        return Vec::new();
    };
    if let Some(error) = &exif.parse_error {
        return vec![(
            "Status".to_owned(),
            format!("Embedded EXIF cannot be parsed ({error})"),
        )];
    }

    let mut rows = Vec::new();
    if let Some(camera) =
        combined_exif_name(exif.camera_make.as_deref(), exif.camera_model.as_deref())
    {
        rows.push(("Camera".to_owned(), camera));
    }
    if let Some(lens) = combined_exif_name(exif.lens_make.as_deref(), exif.lens_model.as_deref()) {
        rows.push(("Lens".to_owned(), lens));
    }
    if let Some(captured_at) = &exif.captured_at {
        rows.push(("Captured".to_owned(), format_exif_datetime(captured_at)));
    }

    if let Some(seconds) = exif.exposure_time_seconds.filter(|value| *value > 0.0) {
        rows.push(("Shutter".to_owned(), format_exposure_time(seconds)));
    }
    if let Some(f_number) = exif.aperture_f_number.filter(|value| *value > 0.0) {
        rows.push((
            "Aperture".to_owned(),
            format!("f/{}", format_decimal(f_number)),
        ));
    }
    if let Some(iso) = exif.iso_speed.filter(|value| *value > 0) {
        rows.push(("ISO".to_owned(), iso.to_string()));
    }
    if let Some(focal_length) = exif.focal_length_mm.filter(|value| *value > 0.0) {
        rows.push((
            "Focal length".to_owned(),
            format!("{} mm", format_decimal(focal_length)),
        ));
    }
    if let Some(bias) = exif.exposure_bias_ev.filter(|value| value.is_finite()) {
        rows.push(("Exposure bias".to_owned(), format!("{bias:+.1} EV")));
    }

    rows
}

fn attribution_metadata_rows(
    exif: Option<&ExifMetadata>,
    xmp: Option<&XmpMetadata>,
) -> Vec<(String, String)> {
    let mut rows = Vec::new();
    if let Some(rating) = xmp_rating_row(xmp) {
        rows.push(rating);
    }
    if let Some(exif) = exif.filter(|metadata| metadata.parse_error.is_none()) {
        if let Some(artist) = &exif.artist {
            rows.push(("Artist".to_owned(), artist.clone()));
        }
        if let Some(copyright) = &exif.copyright {
            rows.push(("Copyright".to_owned(), copyright.clone()));
        }
        if let Some(software) = &exif.software {
            rows.push(("Software".to_owned(), software.clone()));
        }
    }
    rows
}

fn format_exif_datetime(value: &str) -> String {
    exif::DateTime::from_ascii(value.as_bytes())
        .map_or_else(|_| value.to_owned(), |date| date.to_string())
}

fn xmp_rating_row(xmp: Option<&XmpMetadata>) -> Option<(String, String)> {
    let rating = xmp?.rating?;
    let value = if rating.is_sign_negative() {
        "Rejected".to_owned()
    } else if rating.abs() < f64::EPSILON {
        "Unrated".to_owned()
    } else {
        let unit = if (rating - 1.0).abs() < f64::EPSILON {
            "Star"
        } else {
            "Stars"
        };
        format!("{} {unit}", format_decimal(rating))
    };
    Some(("Rating".to_owned(), value))
}

fn combined_exif_name(first: Option<&str>, second: Option<&str>) -> Option<String> {
    match (first, second) {
        (Some(first), Some(second)) if second.starts_with(first) => Some(second.to_owned()),
        (Some(first), Some(second)) => Some(format!("{first} {second}")),
        (Some(value), None) | (None, Some(value)) => Some(value.to_owned()),
        (None, None) => None,
    }
}

fn format_exposure_time(seconds: f64) -> String {
    if seconds < 1.0 {
        let reciprocal = 1.0 / seconds;
        let rounded = reciprocal.round();
        if (reciprocal - rounded).abs() < 0.05 {
            return format!("1/{rounded:.0} s");
        }
    }
    format!("{} s", format_decimal(seconds))
}

fn format_decimal(value: f64) -> String {
    let formatted = format!("{value:.2}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

#[cfg(test)]
mod tests {
    use xl_view::decode::{DecodeLimits, decode_file};

    use super::*;
    use crate::test_utils::fixture_path;

    #[test]
    fn source_encoding_details_are_concise() {
        let sdr = xl_view::decode::SourceColorEncoding::Enumerated {
            color_space: "RGB".to_owned(),
            white_point: "D65".to_owned(),
            primaries: Some("sRGB".to_owned()),
            transfer_function: "sRGB".to_owned(),
        };
        assert_eq!(
            source_encoding_details(&sdr),
            (
                "RGB, sRGB primaries, D65 white".to_owned(),
                "sRGB".to_owned()
            )
        );
        let gray = xl_view::decode::SourceColorEncoding::Enumerated {
            color_space: "Grayscale".to_owned(),
            white_point: "D50".to_owned(),
            primaries: None,
            transfer_function: "PQ".to_owned(),
        };
        assert_eq!(
            source_encoding_details(&gray).0,
            "Grayscale, D50 white".to_owned()
        );
        let icc = xl_view::decode::SourceColorEncoding::Icc {
            color_space: "RGB".to_owned(),
            profile_bytes: 21_012,
        };
        assert_eq!(
            source_encoding_details(&icc),
            (
                "ICC profile, RGB (21012 bytes)".to_owned(),
                "ICC-defined".to_owned()
            )
        );
        assert_eq!(source_range_summary(false, "sRGB"), "SDR (sRGB)");
        assert_eq!(source_range_summary(true, "HLG"), "HDR (HLG)");
        assert_eq!(source_range_summary(true, "PQ"), "HDR (PQ)");
    }

    #[test]
    fn decode_summary_combines_time_and_memory() {
        const MIB: u64 = 1024 * 1024;
        assert_eq!(
            decode_summary(
                DecodeTiming::Measured(Duration::from_millis(611)),
                usize::try_from(16 * MIB).unwrap(),
            ),
            "611.0 ms, 16 MiB"
        );
        assert_eq!(
            decode_summary(DecodeTiming::CacheHit(Duration::from_millis(611)), 0),
            "611.0 ms (cached), 0 MiB"
        );
    }

    #[test]
    fn dimensions_summary_includes_decimal_megapixels() {
        assert_eq!(dimensions_summary(4000, 3000), "4000 x 3000 (12.0 MP)");
        assert_eq!(dimensions_summary(1024, 1024), "1024 x 1024 (1.0 MP)");
    }

    #[test]
    fn embedded_metadata_is_grouped_and_missing_fields_are_omitted() {
        let exif = ExifMetadata {
            aperture_f_number: Some(2.8),
            artist: Some("Ada Example".to_owned()),
            camera_make: Some("ACME".to_owned()),
            camera_model: Some("ACME Photon 1".to_owned()),
            captured_at: Some("2026:07:13 12:34:56".to_owned()),
            copyright: Some("CC0 fixture".to_owned()),
            exposure_bias_ev: Some(-1.0 / 3.0),
            exposure_time_seconds: Some(1.0 / 125.0),
            focal_length_mm: Some(50.0),
            iso_speed: Some(200),
            lens_make: None,
            lens_model: Some("Prime 50".to_owned()),
            parse_error: None,
            software: None,
        };
        assert_eq!(
            capture_metadata_rows(Some(&exif)),
            [
                ("Camera".to_owned(), "ACME Photon 1".to_owned()),
                ("Lens".to_owned(), "Prime 50".to_owned()),
                ("Captured".to_owned(), "2026-07-13 12:34:56".to_owned()),
                ("Shutter".to_owned(), "1/125 s".to_owned()),
                ("Aperture".to_owned(), "f/2.8".to_owned()),
                ("ISO".to_owned(), "200".to_owned()),
                ("Focal length".to_owned(), "50 mm".to_owned()),
                ("Exposure bias".to_owned(), "-0.3 EV".to_owned()),
            ]
        );
        assert_eq!(
            attribution_metadata_rows(Some(&exif), None),
            [
                ("Artist".to_owned(), "Ada Example".to_owned()),
                ("Copyright".to_owned(), "CC0 fixture".to_owned()),
            ]
        );
        assert!(capture_metadata_rows(None).is_empty());
        assert!(attribution_metadata_rows(None, None).is_empty());

        let mut empty = exif.clone();
        empty.aperture_f_number = None;
        empty.artist = None;
        empty.camera_make = None;
        empty.camera_model = None;
        empty.captured_at = None;
        empty.copyright = None;
        empty.exposure_bias_ev = None;
        empty.exposure_time_seconds = None;
        empty.focal_length_mm = None;
        empty.iso_speed = None;
        empty.lens_make = None;
        empty.lens_model = None;
        empty.software = None;
        assert!(capture_metadata_rows(Some(&empty)).is_empty());
        assert!(attribution_metadata_rows(Some(&empty), None).is_empty());

        assert_eq!(format_exif_datetime("not a date"), "not a date".to_owned());
    }

    #[test]
    fn xmp_rating_is_the_first_attribution_row() {
        let xmp = XmpMetadata {
            parse_error: None,
            rating: Some(4.5),
        };
        assert_eq!(
            attribution_metadata_rows(None, Some(&xmp)).first(),
            Some(&("Rating".to_owned(), "4.5 Stars".to_owned()))
        );

        let one_star = XmpMetadata {
            parse_error: None,
            rating: Some(1.0),
        };
        assert_eq!(
            xmp_rating_row(Some(&one_star)),
            Some(("Rating".to_owned(), "1 Star".to_owned()))
        );

        let rejected = XmpMetadata {
            parse_error: None,
            rating: Some(-1.0),
        };
        assert_eq!(
            xmp_rating_row(Some(&rejected)),
            Some(("Rating".to_owned(), "Rejected".to_owned()))
        );

        let unrated = XmpMetadata {
            parse_error: None,
            rating: Some(0.0),
        };
        assert_eq!(
            xmp_rating_row(Some(&unrated)),
            Some(("Rating".to_owned(), "Unrated".to_owned()))
        );
    }

    #[test]
    fn decoded_source_rows_use_the_renderers_peak_with_linear_unit_or_hdr_target() {
        for (name, range, peak, hdr) in [
            (
                "test_pattern-scRGB.jxl",
                "HDR (Linear)",
                "1000 nits (unit 80 nits)",
                true,
            ),
            (
                "linear-bt2020.jxl",
                "SDR (Linear)",
                "203 nits (unit 203 nits, assumed)",
                false,
            ),
            (
                "linear-highlights.jxl",
                "HDR (Linear)",
                "812 nits (unit 203 nits, assumed)",
                true,
            ),
            // Tone mapping still uses the measured overshoot, without an HDR claim.
            (
                "linear-xyb-sdr.jxl",
                "SDR (Linear)",
                "204 nits (unit 203 nits, assumed)",
                false,
            ),
            ("test_pattern-sRGB.jxl", "SDR (sRGB)", "203 nits", false),
            (
                "test_pattern-HLG.jxl",
                "HDR (HLG)",
                "1000 nits (target 1000 nits)",
                true,
            ),
            // PQ and HLG content can exceed or stay far below the target that
            // supports their HDR claim.
            (
                "hdr-P3-PQ.jxl",
                "HDR (PQ)",
                "1531 nits (target 1000 nits)",
                true,
            ),
            (
                "hdr-sRGB-HLG.jxl",
                "HDR (HLG)",
                "211 nits (target 1000 nits)",
                true,
            ),
        ] {
            let path = fixture_path(name);
            let image = decode_file(&path, DecodeLimits::from_memory_ceiling_mib(64)).unwrap();
            let summary = LoadedImageSummary::from_decoded(
                &path,
                &image,
                DecodeTiming::Measured(Duration::ZERO),
            );
            assert_eq!(
                source_range_summary(summary.hdr_source, &summary.source_transfer),
                range
            );
            assert_eq!(summary.source_peak_summary(), peak);
            // The title warning and range row share this same flag.
            assert_eq!(summary.hdr_source, hdr, "{name}");
        }
    }
}
