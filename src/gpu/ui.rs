use std::sync::OnceLock;

use fontdue::{Font, FontSettings, LineMetrics};
use xl_view::color::{HDR_REFERENCE_WHITE_NITS, PQ_PEAK_NITS, SourceIntensityTarget, srgb_eotf};
use xl_view::decode::{HISTOGRAM_BINS_PER_STOP, HISTOGRAM_REFERENCE_WHITE_BIN, Histogram};
use xl_view::units::usize_from_u32;

use crate::APPLICATION_NAME;
use crate::app_icon;

const FONT_BYTES: &[u8] = include_bytes!("../../assets/fonts/AdwaitaSans-Regular.ttf");
const MAX_CHARACTERS_PER_CELL: usize = 80;
const EMPTY_STATE_HELP: &str = r#"Use drag and drop or press "O" to open a file"#;

// UI colors are linear-light BT.709 with straight alpha, where 1.0 is HDR
// reference white. They are not sRGB-encoded: tune them as linear values.
// Only the logo is sRGB-encoded and goes through `srgb_to_linear`.
/// Overlay panel. `EMPTY_STATE_GREY` in presentation.wgsl repeats its grey.
const PANEL_RGBA: [f32; 4] = [0.012, 0.012, 0.012, 0.84];
const SECTION_TITLE_RGB: [f32; 3] = [0.82, 0.86, 0.94];
const SECTION_RULE_RGB: [f32; 3] = [0.20, 0.23, 0.29];
const LABEL_RGB: [f32; 3] = [0.68, 0.72, 0.78];
const VALUE_RGB: [f32; 3] = [0.94, 0.94, 0.94];
const HISTOGRAM_BAR_RGB: [f32; 3] = [0.5, 0.5, 0.5];
const EMPTY_STATE_TITLE_RGB: [f32; 3] = [0.94, 0.95, 0.97];
const EMPTY_STATE_HELP_RGB: [f32; 3] = [0.60, 0.64, 0.70];

pub struct OverlayImage {
    pub height: u32,
    pub pixels: Vec<f32>,
    pub width: u32,
}

#[derive(Debug, PartialEq)]
pub struct OverlaySection {
    pub title: &'static str,
    pub rows: Vec<OverlayRow>,
}

#[derive(Debug, PartialEq)]
#[expect(clippy::large_enum_variant)] // An overlay has a handful of rows.
pub enum OverlayRow {
    /// A label and its value.
    Text(String, String),
    /// A label and a histogram as wide as the value column.
    Histogram {
        label: String,
        bins: Histogram,
        /// Marked with a solid line.
        intensity_target: Option<SourceIntensityTarget>,
    },
}

#[derive(Clone, Copy)]
struct OverlayLayout {
    content_width: usize,
    histogram_height: usize,
    label_width: usize,
    row_ascent: usize,
    row_height: usize,
    title_ascent: usize,
    title_height: usize,
    width: usize,
}

impl OverlayLayout {
    fn height_of(self, row: &OverlayRow) -> usize {
        match row {
            OverlayRow::Text(..) => self.row_height,
            OverlayRow::Histogram { .. } => self.histogram_height,
        }
    }
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
    let line_metrics = font_line_metrics(font, scale.font_px);
    let title_metrics = font_line_metrics(font, scale.title_font_px);
    let row_height = ceil_to_usize(line_metrics.new_line_size).max(1);
    let title_height = ceil_to_usize(title_metrics.new_line_size).max(1);
    let rows = || sections.iter().flat_map(|section| &section.rows);
    let label_width = rows()
        .map(|row| {
            let (OverlayRow::Text(label, _) | OverlayRow::Histogram { label, .. }) = row;
            text_width(font, label, scale.font_px)
        })
        .max()
        .unwrap_or(1);
    let value_width = rows()
        .filter_map(|row| match row {
            OverlayRow::Text(_, value) => Some(text_width(font, value, scale.font_px)),
            // Takes the width of the widest value.
            OverlayRow::Histogram { .. } => None,
        })
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
    let layout = OverlayLayout {
        content_width,
        histogram_height: row_height * 3,
        label_width,
        row_ascent: ceil_to_usize(line_metrics.ascent),
        row_height,
        title_ascent: ceil_to_usize(title_metrics.ascent),
        title_height,
        width,
    };
    let table_height = rows().map(|row| layout.height_of(row)).sum::<usize>();
    let section_height =
        title_height + scale.title_rule_gap + scale.rule_thickness + scale.rule_row_gap;
    let height = scale.padding * 2
        + table_height
        + sections.len() * section_height
        + sections.len().saturating_sub(1) * scale.section_gap;
    let mut pixels = vec![0.0_f32; width * height * 4];
    fill(&mut pixels, PANEL_RGBA);

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
        SECTION_TITLE_RGB,
    );
    y += layout.title_height + scale.title_rule_gap;
    fill_rectangle(
        pixels,
        layout.width,
        (scale.padding, y),
        (layout.content_width, scale.rule_thickness),
        SECTION_RULE_RGB,
    );
    y += scale.rule_thickness + scale.rule_row_gap;

    let value_x = scale.padding + layout.label_width + scale.column_gap;
    for row in &section.rows {
        let (OverlayRow::Text(label, _) | OverlayRow::Histogram { label, .. }) = row;
        let baseline = y + layout.row_ascent;
        draw_text(
            pixels,
            layout.width,
            scale.padding,
            baseline,
            label,
            scale.font_px,
            LABEL_RGB,
        );
        match row {
            OverlayRow::Text(_, value) => draw_text(
                pixels,
                layout.width,
                value_x,
                baseline,
                value,
                scale.font_px,
                VALUE_RGB,
            ),
            OverlayRow::Histogram {
                bins,
                intensity_target,
                ..
            } => draw_histogram(pixels, bins, *intensity_target, scale, layout, (value_x, y)),
        }
        y += layout.height_of(row);
    }
    y
}

/// Draws the bars of a histogram row from `x` to the right edge of the rows,
/// and its markings over them: a dashed line at reference white and at each
/// stop above it, and a solid line at the end of the intensity target's bin.
#[expect(clippy::cast_precision_loss)] // A few hundred pixels and 125 bins.
fn draw_histogram(
    pixels: &mut [f32],
    bins: &Histogram,
    intensity_target: Option<SourceIntensityTarget>,
    scale: &UiScale,
    layout: OverlayLayout,
    (x, y): (usize, usize),
) {
    let width = scale.padding + layout.content_width - x;
    let top = y + scale.histogram_inset;
    let height = layout.histogram_height - 2 * scale.histogram_inset;
    // The axis ends at the peak of PQ, inside the last bin.
    let end = histogram_position(f64::from(PQ_PEAK_NITS));
    let x_at = |position: f32| x + ceil_to_usize(width as f32 * (position / end).clamp(0.0, 1.0));
    let largest = bins.iter().copied().max().unwrap_or(0);
    for (bin, count) in bins.iter().enumerate() {
        let bar_height = histogram_bar_height(*count, largest, height);
        let left = x_at(bin as f32);
        fill_rectangle(
            pixels,
            layout.width,
            (left, top + height - bar_height),
            (x_at((bin + 1) as f32) - left, bar_height),
            HISTOGRAM_BAR_RGB,
        );
    }
    let mut draw_line = |position: f32, dash: usize| {
        // A line at the end of the axis stays inside the histogram.
        let line_x = x_at(position).min(x + width - scale.rule_thickness);
        for dash_y in (0..height).step_by(dash * 2) {
            fill_rectangle(
                pixels,
                layout.width,
                (line_x, top + dash_y),
                (scale.rule_thickness, dash.min(height - dash_y)),
                SECTION_RULE_RGB,
            );
        }
    };
    for bin in (HISTOGRAM_REFERENCE_WHITE_BIN..bins.len()).step_by(HISTOGRAM_BINS_PER_STOP) {
        draw_line(bin as f32, scale.histogram_dash);
    }
    if let Some(target) = intensity_target {
        // The upper edge of the target's bin, where the bars of content up to
        // the target end. A dash as long as the line draws it solid.
        draw_line(histogram_position(target.nits()).ceil(), height);
    }
}

/// Where a luminance lies on the axis of the histogram, in bins.
#[expect(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // A position on an axis of 125 bins.
fn histogram_position(nits: f64) -> f32 {
    let stops = (nits / f64::from(HDR_REFERENCE_WHITE_NITS)).log2() as f32;
    HISTOGRAM_REFERENCE_WHITE_BIN as f32 + HISTOGRAM_BINS_PER_STOP as f32 * stops
}

/// Scales by the square root of the count, which keeps small bins readable.
/// Rounding up shows every bin that is not empty.
#[expect(clippy::cast_precision_loss)] // A bar is a few dozen pixels tall.
fn histogram_bar_height(count: u64, largest: u64, height: usize) -> usize {
    if count == 0 {
        return 0;
    }
    ceil_to_usize((count as f32 / largest as f32).sqrt() * height as f32).min(height)
}

pub fn render_empty_state(scale_factor: f64) -> OverlayImage {
    let scale = empty_state_scale(scale_factor);
    let font = overlay_font();
    let title = format!("{APPLICATION_NAME} {}", env!("CARGO_PKG_VERSION"));
    let title_width = text_width(font, &title, scale.title_font_px);
    let help_width = text_width(font, EMPTY_STATE_HELP, scale.help_font_px);
    let title_metrics = font_line_metrics(font, scale.title_font_px);
    let help_metrics = font_line_metrics(font, scale.help_font_px);
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
        EMPTY_STATE_TITLE_RGB,
    );

    let help_y = title_y + title_height + scale.title_help_gap;
    draw_text(
        &mut pixels,
        width,
        (width - help_width) / 2,
        help_y + ceil_to_usize(help_metrics.ascent),
        EMPTY_STATE_HELP,
        scale.help_font_px,
        EMPTY_STATE_HELP_RGB,
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

#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // The scale factor is clamped to 1..=3 before producing small positive pixel sizes.
fn empty_state_scale(scale_factor: f64) -> EmptyStateScale {
    let factor = scale_factor.clamp(1.0, 3.0) as f32;
    let pixels = |base: f32| (base * factor).round() as usize;
    EmptyStateScale {
        help_font_px: 16.0 * factor,
        horizontal_padding: pixels(8.0),
        logo_px: (152.0 * factor).round() as u32,
        logo_title_gap: pixels(24.0),
        title_font_px: 28.0 * factor,
        title_help_gap: pixels(10.0),
    }
}

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

#[expect(clippy::cast_possible_truncation)] // Eight-bit UI input is converted by the shared f64 reference before RGBA16F storage.
fn srgb_to_linear(encoded: f32) -> f32 {
    srgb_eotf(f64::from(encoded)) as f32
}

struct UiScale {
    column_gap: usize,
    font_px: f32,
    histogram_dash: usize,
    histogram_inset: usize,
    padding: usize,
    rule_row_gap: usize,
    rule_thickness: usize,
    section_gap: usize,
    title_font_px: f32,
    title_rule_gap: usize,
}

#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // The scale factor is clamped to 1..=3 before producing small positive pixel sizes.
fn ui_scale(scale_factor: f64) -> UiScale {
    let factor = scale_factor.clamp(1.0, 3.0) as f32;
    let pixels = |base: f32| (base * factor).round() as usize;
    UiScale {
        column_gap: pixels(24.0),
        font_px: 17.0 * factor,
        histogram_dash: pixels(3.0),
        histogram_inset: pixels(4.0),
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

fn font_line_metrics(font: &Font, font_px: f32) -> LineMetrics {
    font.horizontal_line_metrics(font_px)
        .expect("Adwaita Sans provides horizontal line metrics")
}

/// The characters drawn for one cell, each with the kerning applied before it.
fn kerned_characters<'a>(
    font: &'a Font,
    text: &'a str,
    font_px: f32,
) -> impl Iterator<Item = (char, f32)> + 'a {
    let mut previous = None;
    text.chars()
        .take(MAX_CHARACTERS_PER_CELL)
        .map(move |character| {
            let kern = previous
                .and_then(|previous| font.horizontal_kern(previous, character, font_px))
                .unwrap_or(0.0);
            previous = Some(character);
            (character, kern)
        })
}

fn text_width(font: &Font, text: &str, font_px: f32) -> usize {
    let width = kerned_characters(font, text, font_px).fold(0.0_f32, |width, (character, kern)| {
        width + kern + font.metrics(character, font_px).advance_width
    });
    ceil_to_usize(width).max(1)
}

#[expect(clippy::cast_precision_loss)] // Coordinates come from 3x-clamped application layouts with text capped at 80 characters per cell.
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
    for (character, kern) in kerned_characters(font, text, font_px) {
        cursor += kern;
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

/// Recolors a rectangle and keeps the alpha of the panel under it.
fn fill_rectangle(
    pixels: &mut [f32],
    canvas_width: usize,
    (x, y): (usize, usize),
    (width, height): (usize, usize),
    rgb: [f32; 3],
) {
    let canvas_height = pixels.len() / 4 / canvas_width;
    for pixel_y in y..y.saturating_add(height).min(canvas_height) {
        for pixel_x in x..x.saturating_add(width).min(canvas_width) {
            let pixel = &mut pixels[(pixel_y * canvas_width + pixel_x) * 4..][..4];
            pixel[..3].copy_from_slice(&rgb);
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

#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Layout measurements are finite and non-negative before conversion.
fn ceil_to_usize(value: f32) -> usize {
    value.max(0.0).ceil() as usize
}

#[expect(clippy::cast_possible_truncation)] // Glyph coordinates remain within practical UI dimensions.
fn floor_to_i32(value: f32) -> i32 {
    value.floor() as i32
}

#[cfg(test)]
mod tests {
    use xl_view::decode::HISTOGRAM_BINS;

    use super::*;

    #[test]
    fn empty_state_background_matches_the_overlay_panel() {
        let shader = include_str!("shaders/presentation.wgsl");
        let expected = format!("const EMPTY_STATE_GREY: f32 = {:?};", PANEL_RGBA[0]);
        assert!(
            shader.contains(&expected),
            "shader must contain `{expected}` to match the panel grey",
        );
    }

    fn text(label: &str, value: &str) -> OverlayRow {
        OverlayRow::Text(label.to_owned(), value.to_owned())
    }

    fn sections() -> Vec<OverlaySection> {
        vec![
            OverlaySection {
                title: "IMAGE",
                rows: vec![text("File", "photo.jxl"), text("Dimensions", "4000 x 3000")],
            },
            OverlaySection {
                title: "SOURCE",
                rows: vec![text("Range", "HDR (PQ transfer)")],
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
        let [red, green, blue] = SECTION_RULE_RGB;
        let rule = [red, green, blue, PANEL_RGBA[3]];
        assert!(image.pixels.chunks_exact(4).any(|pixel| pixel == rule));
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
    fn histogram_row_is_taller_than_a_text_row_and_fills_the_value_column() {
        let render = |row| {
            let mut sections = sections();
            sections[1].rows.push(row);
            render_text_overlay(&sections, 1.0)
        };
        let histogram = |bins, intensity_target| {
            render(OverlayRow::Histogram {
                label: "Histogram".to_owned(),
                bins,
                intensity_target,
            })
        };
        let count = |image: &OverlayImage, rgb: [f32; 3]| {
            let pixels = image.pixels.chunks_exact(4);
            pixels.filter(|pixel| pixel[..3] == rgb).count()
        };
        // Bar pixels in one column of the panel.
        let bar_height = |image: &OverlayImage, x: usize| {
            let rows = image.pixels.chunks_exact(usize_from_u32(image.width) * 4);
            rows.filter(|row| row[x * 4..][..3] == HISTOGRAM_BAR_RGB)
                .count()
        };
        let text_row = render(text("Histogram", "1000 nits"));
        // The darkest bins, the one of 1,000 nits and the brightest one, away
        // from the dashed lines.
        let mut bins = [0; HISTOGRAM_BINS];
        bins[0] = 4;
        bins[1] = 1;
        bins[HISTOGRAM_REFERENCE_WHITE_BIN + 18] = 4;
        bins[HISTOGRAM_BINS - 1] = 4;
        let plain = histogram(bins, None);
        let scale = ui_scale(1.0);
        let width = usize_from_u32(plain.width);

        assert!(plain.height > text_row.height);
        // The histogram takes the width of the values and adds none.
        assert_eq!(plain.width, text_row.width);
        assert_eq!(count(&text_row, HISTOGRAM_BAR_RGB), 0);
        // The last bar ends at the right edge of the rows.
        let tallest = bar_height(&plain, width - scale.padding - 1);
        assert_ne!(tallest, 0);
        assert_eq!(bar_height(&plain, width - scale.padding), 0);
        let first = (0..width).find(|x| bar_height(&plain, *x) > 0).unwrap();
        assert_eq!(bar_height(&plain, first), tallest);
        // A quarter of the largest count is half as tall.
        let second = (first..width)
            .find(|x| bar_height(&plain, *x) < tallest)
            .unwrap();
        assert_eq!(bar_height(&plain, second), tallest.div_ceil(2));
        // Beside the two section rules, the markings are six dashed lines and
        // a solid line for the intensity target.
        let section_rules = 2 * (width - 2 * scale.padding) * scale.rule_thickness;
        let dashed = count(&plain, SECTION_RULE_RGB) - section_rules;
        let solid = tallest * scale.rule_thickness;
        assert!(2 * solid < dashed && dashed < 4 * solid);
        let marked = histogram(bins, SourceIntensityTarget::new(1_000.0));
        assert_eq!(
            count(&marked, SECTION_RULE_RGB) - section_rules,
            dashed + solid
        );
        // The solid line stands at the end of the bar of the target's bin and
        // covers none of it.
        assert_eq!(
            count(&marked, HISTOGRAM_BAR_RGB),
            count(&plain, HISTOGRAM_BAR_RGB)
        );
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
