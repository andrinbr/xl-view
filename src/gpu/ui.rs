use std::sync::OnceLock;

use fontdue::{Font, FontSettings};
use xl_view::color::srgb_eotf;

use crate::APPLICATION_NAME;
use crate::app_icon;
use crate::units::usize_from_u32;

const FONT_BYTES: &[u8] = include_bytes!("../../assets/fonts/AdwaitaSans-Regular.ttf");
const MAX_CHARACTERS_PER_CELL: usize = 80;
const EMPTY_STATE_HELP: &str = r#"Use drag and drop or press "O" to open a file"#;

pub struct OverlayImage {
    pub height: u32,
    pub pixels: Vec<f32>,
    pub width: u32,
}

#[derive(Debug, Eq, PartialEq)]
pub struct OverlaySection {
    pub title: &'static str,
    pub rows: Vec<(String, String)>,
}

#[derive(Clone, Copy)]
struct OverlayLayout {
    content_width: usize,
    label_width: usize,
    row_ascent: usize,
    row_height: usize,
    title_ascent: usize,
    title_height: usize,
    width: usize,
}

pub fn render_text_overlay(sections: &[OverlaySection], scale_factor: f64) -> OverlayImage {
    let sections = sections
        .iter()
        .filter(|section| !section.rows.is_empty())
        .collect::<Vec<_>>();
    if sections.is_empty() {
        return OverlayImage {
            height: 1,
            pixels: vec![0.0; 4],
            width: 1,
        };
    }

    let font = overlay_font();
    let scale = ui_scale(scale_factor);
    let line_metrics = font
        .horizontal_line_metrics(scale.font_px)
        .expect("Adwaita Sans provides horizontal line metrics");
    let title_metrics = font
        .horizontal_line_metrics(scale.title_font_px)
        .expect("Adwaita Sans provides horizontal line metrics");
    let row_height = ceil_to_usize(line_metrics.new_line_size).max(1);
    let title_height = ceil_to_usize(title_metrics.new_line_size).max(1);
    let label_width = sections
        .iter()
        .flat_map(|section| &section.rows)
        .map(|(label, _)| text_width(font, label, scale.font_px))
        .max()
        .unwrap_or(1);
    let value_width = sections
        .iter()
        .flat_map(|section| &section.rows)
        .map(|(_, value)| text_width(font, value, scale.font_px))
        .max()
        .unwrap_or(1);
    let table_width = label_width + scale.column_gap + value_width;
    let title_width = sections
        .iter()
        .map(|section| text_width(font, section.title, scale.title_font_px))
        .max()
        .unwrap_or(1);
    let content_width = table_width.max(title_width);
    let width = scale.padding * 2 + content_width;
    let table_height = sections
        .iter()
        .map(|section| section.rows.len() * row_height)
        .sum::<usize>();
    let section_height =
        title_height + scale.title_rule_gap + scale.rule_thickness + scale.rule_row_gap;
    let height = scale.padding * 2
        + table_height
        + sections.len() * section_height
        + sections.len().saturating_sub(1) * scale.section_gap;
    let mut pixels = vec![0.0_f32; width * height * 4];
    fill(&mut pixels, [0.012, 0.012, 0.012, 0.84]);

    let layout = OverlayLayout {
        content_width,
        label_width,
        row_ascent: ceil_to_usize(line_metrics.ascent),
        row_height,
        title_ascent: ceil_to_usize(title_metrics.ascent),
        title_height,
        width,
    };
    let mut y = scale.padding;
    for (section_index, section) in sections.iter().enumerate() {
        if section_index > 0 {
            y += scale.section_gap;
        }
        y = draw_overlay_section(&mut pixels, section, &scale, layout, y);
    }

    OverlayImage {
        height: u32::try_from(height)
            .expect("the application-generated metadata overlay height must fit u32"),
        pixels,
        width: u32::try_from(width)
            .expect("the application-generated metadata overlay width must fit u32"),
    }
}

fn draw_overlay_section(
    pixels: &mut [f32],
    section: &OverlaySection,
    scale: &UiScale,
    layout: OverlayLayout,
    mut y: usize,
) -> usize {
    draw_text(
        pixels,
        layout.width,
        scale.padding,
        y + layout.title_ascent,
        section.title,
        scale.title_font_px,
        [0.82, 0.86, 0.94],
    );
    y += layout.title_height + scale.title_rule_gap;
    draw_horizontal_rule(
        pixels,
        layout.width,
        scale.padding,
        y,
        layout.content_width,
        scale.rule_thickness,
    );
    y += scale.rule_thickness + scale.rule_row_gap;

    for (row_index, (label, value)) in section.rows.iter().enumerate() {
        let baseline = y + row_index * layout.row_height + layout.row_ascent;
        draw_text(
            pixels,
            layout.width,
            scale.padding,
            baseline,
            label,
            scale.font_px,
            [0.68, 0.72, 0.78],
        );
        draw_text(
            pixels,
            layout.width,
            scale.padding + layout.label_width + scale.column_gap,
            baseline,
            value,
            scale.font_px,
            [0.94, 0.94, 0.94],
        );
    }
    y + section.rows.len() * layout.row_height
}

pub fn render_empty_state(scale_factor: f64) -> OverlayImage {
    let scale = empty_state_scale(scale_factor);
    let font = overlay_font();
    let title = format!("{APPLICATION_NAME} {}", env!("CARGO_PKG_VERSION"));
    let title_width = text_width(font, &title, scale.title_font_px);
    let help_width = text_width(font, EMPTY_STATE_HELP, scale.help_font_px);
    let title_metrics = font
        .horizontal_line_metrics(scale.title_font_px)
        .expect("Adwaita Sans provides horizontal line metrics");
    let help_metrics = font
        .horizontal_line_metrics(scale.help_font_px)
        .expect("Adwaita Sans provides horizontal line metrics");
    let title_height = ceil_to_usize(title_metrics.new_line_size).max(1);
    let help_height = ceil_to_usize(help_metrics.new_line_size).max(1);
    let logo_px = usize_from_u32(scale.logo_px);
    let content_width = logo_px.max(title_width).max(help_width);
    let width = content_width + scale.horizontal_padding * 2;
    let height = logo_px + scale.logo_title_gap + title_height + scale.title_help_gap + help_height;
    let mut pixels = vec![0.0_f32; width * height * 4];

    draw_svg_logo(&mut pixels, width, (width - logo_px) / 2, 0, scale.logo_px);

    let title_y = logo_px + scale.logo_title_gap;
    draw_text(
        &mut pixels,
        width,
        (width - title_width) / 2,
        title_y + ceil_to_usize(title_metrics.ascent),
        &title,
        scale.title_font_px,
        [0.94, 0.95, 0.97],
    );

    let help_y = title_y + title_height + scale.title_help_gap;
    draw_text(
        &mut pixels,
        width,
        (width - help_width) / 2,
        help_y + ceil_to_usize(help_metrics.ascent),
        EMPTY_STATE_HELP,
        scale.help_font_px,
        [0.60, 0.64, 0.70],
    );

    OverlayImage {
        height: u32::try_from(height)
            .expect("the 3x-clamped empty-state layout height must fit u32"),
        pixels,
        width: u32::try_from(width).expect("the 3x-clamped empty-state layout width must fit u32"),
    }
}

struct EmptyStateScale {
    help_font_px: f32,
    horizontal_padding: usize,
    logo_px: u32,
    logo_title_gap: usize,
    title_font_px: f32,
    title_help_gap: usize,
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // The scale factor is clamped to 1..=3 before producing small positive pixel sizes.
fn empty_state_scale(scale_factor: f64) -> EmptyStateScale {
    let factor = scale_factor.clamp(1.0, 3.0) as f32;
    EmptyStateScale {
        help_font_px: 16.0 * factor,
        horizontal_padding: (8.0 * factor).round() as usize,
        logo_px: (152.0 * factor).round() as u32,
        logo_title_gap: (24.0 * factor).round() as usize,
        title_font_px: 28.0 * factor,
        title_help_gap: (10.0 * factor).round() as usize,
    }
}

#[allow(clippy::cast_precision_loss)] // The 3x-clamped empty-state scale produces a logo of at most 456 pixels.
fn draw_svg_logo(pixels: &mut [f32], canvas_width: usize, x: usize, y: usize, logo_px: u32) {
    let icon_pixels = app_icon::rasterize(logo_px);

    let canvas_height = pixels.len() / 4 / canvas_width;
    for (source_y, row) in icon_pixels
        .chunks_exact(usize_from_u32(logo_px) * 4)
        .enumerate()
    {
        let destination_y = y + source_y;
        if destination_y >= canvas_height {
            break;
        }
        for (source_x, source) in row.chunks_exact(4).enumerate() {
            let destination_x = x + source_x;
            if destination_x >= canvas_width {
                break;
            }
            let alpha = f32::from(source[3]) / 255.0;
            let destination =
                &mut pixels[(destination_y * canvas_width + destination_x) * 4..][..4];
            if alpha == 0.0 {
                destination.fill(0.0);
                continue;
            }
            for channel in 0..3 {
                let encoded = f32::from(source[channel]) / 255.0;
                destination[channel] = srgb_to_linear(encoded);
            }
            destination[3] = alpha;
        }
    }
}

#[allow(clippy::cast_possible_truncation)] // Eight-bit UI input is converted by the shared f64 reference before RGBA16F storage.
fn srgb_to_linear(encoded: f32) -> f32 {
    srgb_eotf(f64::from(encoded)) as f32
}

struct UiScale {
    column_gap: usize,
    font_px: f32,
    padding: usize,
    rule_row_gap: usize,
    rule_thickness: usize,
    section_gap: usize,
    title_font_px: f32,
    title_rule_gap: usize,
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // The scale factor is clamped to 1..=3 before producing small positive pixel sizes.
fn ui_scale(scale_factor: f64) -> UiScale {
    let factor = scale_factor.clamp(1.0, 3.0) as f32;
    let pixels = |base: f32| (base * factor).round() as usize;
    UiScale {
        column_gap: pixels(24.0),
        font_px: 17.0 * factor,
        padding: pixels(14.0),
        rule_row_gap: pixels(7.0),
        rule_thickness: pixels(1.0).max(1),
        section_gap: pixels(15.0),
        title_font_px: 13.0 * factor,
        title_rule_gap: pixels(4.0),
    }
}

fn overlay_font() -> &'static Font {
    static FONT: OnceLock<Font> = OnceLock::new();
    FONT.get_or_init(|| {
        Font::from_bytes(FONT_BYTES, FontSettings::default())
            .expect("the bundled Adwaita Sans font must remain valid")
    })
}

fn text_width(font: &Font, text: &str, font_px: f32) -> usize {
    let mut width = 0.0_f32;
    let mut previous = None;
    for character in text.chars().take(MAX_CHARACTERS_PER_CELL) {
        if let Some(previous) = previous {
            width += font
                .horizontal_kern(previous, character, font_px)
                .unwrap_or(0.0);
        }
        width += font.metrics(character, font_px).advance_width;
        previous = Some(character);
    }
    ceil_to_usize(width).max(1)
}

#[allow(clippy::cast_precision_loss)] // Coordinates come from 3x-clamped application layouts with text capped at 80 characters per cell.
fn draw_text(
    pixels: &mut [f32],
    canvas_width: usize,
    x: usize,
    baseline: usize,
    text: &str,
    font_px: f32,
    text_rgb: [f32; 3],
) {
    let font = overlay_font();
    let canvas_height = pixels.len() / 4 / canvas_width;
    let mut cursor = x as f32;
    let mut previous = None;
    for character in text.chars().take(MAX_CHARACTERS_PER_CELL) {
        if let Some(previous) = previous {
            cursor += font
                .horizontal_kern(previous, character, font_px)
                .unwrap_or(0.0);
        }
        let (metrics, bitmap) = font.rasterize(character, font_px);
        let glyph_x = floor_to_i32(cursor) + metrics.xmin;
        let glyph_y = i32::try_from(baseline).unwrap_or(i32::MAX)
            - metrics.ymin
            - i32::try_from(metrics.height).unwrap_or(i32::MAX);
        blend_rasterized_glyph(
            pixels,
            (canvas_width, canvas_height),
            (glyph_x, glyph_y),
            &metrics,
            &bitmap,
            text_rgb,
        );
        cursor += metrics.advance_width;
        previous = Some(character);
    }
}

fn blend_rasterized_glyph(
    pixels: &mut [f32],
    (canvas_width, canvas_height): (usize, usize),
    (glyph_x, glyph_y): (i32, i32),
    metrics: &fontdue::Metrics,
    bitmap: &[u8],
    text_rgb: [f32; 3],
) {
    for bitmap_y in 0..metrics.height {
        for bitmap_x in 0..metrics.width {
            let coverage = f32::from(bitmap[bitmap_y * metrics.width + bitmap_x]) / 255.0;
            if coverage == 0.0 {
                continue;
            }
            let pixel_x = glyph_x + i32::try_from(bitmap_x).unwrap_or(i32::MAX);
            let pixel_y = glyph_y + i32::try_from(bitmap_y).unwrap_or(i32::MAX);
            let (Ok(pixel_x), Ok(pixel_y)) = (usize::try_from(pixel_x), usize::try_from(pixel_y))
            else {
                continue;
            };
            if pixel_x >= canvas_width || pixel_y >= canvas_height {
                continue;
            }
            blend_text_pixel(
                &mut pixels[(pixel_y * canvas_width + pixel_x) * 4..][..4],
                text_rgb,
                coverage,
            );
        }
    }
}

fn fill(pixels: &mut [f32], rgba: [f32; 4]) {
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.copy_from_slice(&rgba);
    }
}

fn draw_horizontal_rule(
    pixels: &mut [f32],
    canvas_width: usize,
    x: usize,
    y: usize,
    rule_width: usize,
    thickness: usize,
) {
    let canvas_height = pixels.len() / 4 / canvas_width;
    for pixel_y in y..y.saturating_add(thickness).min(canvas_height) {
        for pixel_x in x..x.saturating_add(rule_width).min(canvas_width) {
            let pixel = &mut pixels[(pixel_y * canvas_width + pixel_x) * 4..][..4];
            pixel[..3].copy_from_slice(&[0.20, 0.23, 0.29]);
        }
    }
}

fn blend_text_pixel(pixel: &mut [f32], text_rgb: [f32; 3], coverage: f32) {
    let background_alpha = pixel[3];
    let output_alpha = coverage + background_alpha * (1.0 - coverage);
    for channel in 0..3 {
        let premultiplied =
            text_rgb[channel] * coverage + pixel[channel] * background_alpha * (1.0 - coverage);
        pixel[channel] = premultiplied / output_alpha;
    }
    pixel[3] = output_alpha;
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Layout measurements are finite and non-negative before conversion.
fn ceil_to_usize(value: f32) -> usize {
    value.max(0.0).ceil() as usize
}

#[allow(clippy::cast_possible_truncation)] // Glyph coordinates remain within practical UI dimensions.
fn floor_to_i32(value: f32) -> i32 {
    value.floor() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sections() -> Vec<OverlaySection> {
        vec![
            OverlaySection {
                title: "IMAGE",
                rows: vec![
                    ("File".to_owned(), "photo.jxl".to_owned()),
                    ("Dimensions".to_owned(), "4000 x 3000".to_owned()),
                ],
            },
            OverlaySection {
                title: "SOURCE",
                rows: vec![("Range".to_owned(), "HDR (PQ)".to_owned())],
            },
        ]
    }

    #[test]
    fn empty_overlay_is_a_transparent_placeholder() {
        let image = render_text_overlay(&[], 1.0);
        assert_eq!((image.width, image.height), (1, 1));
        assert_eq!(image.pixels, [0.0; 4]);
    }

    #[test]
    fn table_overlay_contains_antialiased_text() {
        let image = render_text_overlay(&sections(), 1.0);
        assert!(image.width > 200);
        assert!(image.height > 100);
        assert!(
            image
                .pixels
                .chunks_exact(4)
                .any(|pixel| pixel == [0.20, 0.23, 0.29, 0.84])
        );
        assert!(image.pixels.chunks_exact(4).any(|pixel| pixel[3] > 0.99));
        assert!(
            image
                .pixels
                .chunks_exact(4)
                .any(|pixel| pixel[3] > 0.85 && pixel[3] < 0.99)
        );
    }

    #[test]
    fn overlay_scales_for_hidpi() {
        let normal = render_text_overlay(&sections(), 1.0);
        let hidpi = render_text_overlay(&sections(), 2.0);
        assert_eq!(hidpi.width, normal.width * 2);
        assert_eq!(hidpi.height, normal.height * 2);
    }

    #[test]
    fn empty_sections_are_not_rendered() {
        let empty = [OverlaySection {
            title: "CAPTURE",
            rows: Vec::new(),
        }];
        let image = render_text_overlay(&empty, 1.0);
        assert_eq!((image.width, image.height), (1, 1));
        assert_eq!(image.pixels, [0.0; 4]);
    }

    #[test]
    fn empty_state_contains_the_packaged_logo_and_text() {
        let image = render_empty_state(1.0);
        let scale = empty_state_scale(1.0);
        let font = overlay_font();
        let title = format!("{APPLICATION_NAME} {}", env!("CARGO_PKG_VERSION"));
        let title_width = text_width(font, &title, scale.title_font_px);
        let help_width = text_width(font, EMPTY_STATE_HELP, scale.help_font_px);
        let title_height = ceil_to_usize(
            font.horizontal_line_metrics(scale.title_font_px)
                .unwrap()
                .new_line_size,
        );
        let help_height = ceil_to_usize(
            font.horizontal_line_metrics(scale.help_font_px)
                .unwrap()
                .new_line_size,
        );
        let logo_height = usize::try_from(scale.logo_px).unwrap();
        let required_width =
            logo_height.max(title_width).max(help_width) + scale.horizontal_padding * 2;
        let title_y = logo_height + scale.logo_title_gap;
        let help_y = title_y + title_height + scale.title_help_gap;
        let required_height = help_y + help_height;
        let width = usize::try_from(image.width).unwrap();
        let height = usize::try_from(image.height).unwrap();
        let row_stride = width * 4;

        assert!(width >= required_width);
        assert!(height >= required_height);
        assert!(image.pixels.chunks_exact(4).any(|pixel| pixel[3] == 0.0));
        assert!(
            image.pixels[..logo_height * row_stride]
                .chunks_exact(4)
                .any(|pixel| pixel[3] > 0.0)
        );
        assert!(
            image.pixels[title_y * row_stride..help_y * row_stride]
                .chunks_exact(4)
                .any(|pixel| pixel[3] > 0.0)
        );
        assert!(
            image.pixels[help_y * row_stride..required_height * row_stride]
                .chunks_exact(4)
                .any(|pixel| pixel[3] > 0.0)
        );
    }

    #[test]
    fn empty_state_scales_for_hidpi() {
        let normal = render_empty_state(1.0);
        let hidpi = render_empty_state(2.0);
        assert!(hidpi.width.abs_diff(normal.width * 2) <= 1);
        assert!(hidpi.height.abs_diff(normal.height * 2) <= 1);
    }
}
