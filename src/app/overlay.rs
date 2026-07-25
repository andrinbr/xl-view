use std::path::Path;
use std::time::Duration;

use xl_view::decode::DecodedImage;
use xl_view::metadata::{ExifMetadata, XmpMetadata};

use super::files::file_name;
use crate::units::format_mib;

#[derive(Debug)]
pub(super) struct LoadedImageSummary {
    pub(super) decode_timing: DecodeTiming,
    pub(super) dimensions: (u32, u32),
    pub(super) exif: Option<ExifMetadata>,
    pub(super) file_name: String,
    pub(super) hdr_source: bool,
    pub(super) memory_bytes: usize,
    pub(super) source_color_space: String,
    pub(super) source_intensity_nits: f32,
    pub(super) source_transfer: String,
    pub(super) xmp: Option<XmpMetadata>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DecodeTiming {
    Measured(Duration),
    CacheHit(Duration),
}

impl LoadedImageSummary {
    pub(super) fn from_decoded(
        path: Option<&Path>,
        image: &DecodedImage,
        decode_timing: DecodeTiming,
    ) -> Self {
        let (source_color_space, source_transfer) =
            source_encoding_details(&image.metadata.color_encoding);
        Self {
            decode_timing,
            dimensions: (image.width, image.height),
            file_name: path.map_or_else(|| "image.jxl".to_owned(), file_name),
            exif: image.metadata.exif.clone(),
            hdr_source: image.source_dynamic_range.is_hdr(),
            memory_bytes: image.memory_cost_bytes,
            source_color_space,
            source_transfer,
            source_intensity_nits: image.metadata.tone_mapping.intensity_target_nits,
            xmp: image.metadata.xmp.clone(),
        }
    }
}

pub(super) fn source_encoding_details(
    encoding: &xl_view::decode::SourceColorEncoding,
) -> (String, String) {
    match encoding {
        xl_view::decode::SourceColorEncoding::Enumerated {
            colour_space,
            white_point,
            primaries,
            transfer_function,
        } => (
            format!("{colour_space}, {primaries} primaries, {white_point} white"),
            transfer_function.clone(),
        ),
        xl_view::decode::SourceColorEncoding::Icc {
            colour_space,
            profile_bytes,
        } => (
            format!("ICC {colour_space} ({profile_bytes} profile bytes)"),
            "ICC-defined".to_owned(),
        ),
    }
}

pub(super) fn source_range_summary(hdr: bool, transfer_function: &str) -> String {
    let range = if hdr { "HDR" } else { "SDR" };
    format!("{range} ({transfer_function})")
}

pub(super) fn decode_summary(decode_timing: DecodeTiming, decoded_bytes: usize) -> String {
    let timing = match decode_timing {
        DecodeTiming::Measured(duration) => {
            format!("{:.1} ms", duration.as_secs_f64() * 1_000.0)
        }
        DecodeTiming::CacheHit(duration) => {
            format!("{:.1} ms (cached)", duration.as_secs_f64() * 1_000.0)
        }
    };
    format!(
        "{timing}, {} MiB",
        format_mib(u64::try_from(decoded_bytes).unwrap_or(u64::MAX)),
    )
}

pub(super) fn dimensions_summary(width: u32, height: u32) -> String {
    let megapixels = f64::from(width) * f64::from(height) / 1_000_000.0;
    format!("{width} x {height} ({megapixels:.1} MP)")
}

pub(super) fn capture_metadata_rows(exif: Option<&ExifMetadata>) -> Vec<(String, String)> {
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

pub(super) fn attribution_metadata_rows(
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

pub(super) fn format_exif_datetime(value: &str) -> String {
    exif::DateTime::from_ascii(value.as_bytes())
        .map_or_else(|_| value.to_owned(), |date| date.to_string())
}

pub(super) fn xmp_rating_row(xmp: Option<&XmpMetadata>) -> Option<(String, String)> {
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
