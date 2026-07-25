use super::*;

fn distinct_parameters() -> PresentationParams {
    PresentationParams {
        mode: 1,
        encode_srgb: true,
        dither_bits: 3,
        content_mode: 4,
        hdr_reference_white_nits: 5.0,
        source_peak_nits: 6.0,
        output_peak_nits: 7.0,
        exposure_stops: 8.0,
        viewport_width: 9,
        viewport_height: 10,
        image_width: 11,
        image_height: 12,
        ui_white_nits: 13.0,
        view_center_x: 14.0,
        view_center_y: 15.0,
        view_scale: 16.0,
        background_mode: 17,
        center_ui: true,
        source_dynamic_range: 19,
        tiled_image: true,
        tile_size: 22,
        tile_columns: 23,
        tile_gutter: 24,
        resampled_viewport: true,
    }
}

#[test]
fn parameters_serialize_in_wgsl_word_order() {
    assert_eq!(
        distinct_parameters().to_words(),
        [
            1,                  // 0: mode
            1,                  // 1: encode_srgb
            3,                  // 2: dither_bits
            4,                  // 3: content_mode
            5.0_f32.to_bits(),  // 4: hdr_reference_white_nits
            6.0_f32.to_bits(),  // 5: source_peak_nits
            7.0_f32.to_bits(),  // 6: output_peak_nits
            8.0_f32.to_bits(),  // 7: exposure_stops
            9,                  // 8: viewport_width
            10,                 // 9: viewport_height
            11,                 // 10: image_width
            12,                 // 11: image_height
            13.0_f32.to_bits(), // 12: ui_white_nits
            14.0_f32.to_bits(), // 13: view_center_x
            15.0_f32.to_bits(), // 14: view_center_y
            16.0_f32.to_bits(), // 15: view_scale
            17,                 // 16: background_mode
            1,                  // 17: center_ui
            19,                 // 18: source_dynamic_range
            0,                  // 19: _padding_0
            1,                  // 20: tiled_image
            22,                 // 21: tile_size
            23,                 // 22: tile_columns
            24,                 // 23: tile_gutter
            1,                  // 24: resampled_viewport
            0,                  // 25: _padding_1
            0,                  // 26: _padding_2
            0,                  // 27: _padding_3
        ]
    );
}

#[test]
fn rust_serializer_tracks_the_wgsl_field_sequence() {
    let shader = include_str!("../shaders/presentation.wgsl");
    let (_, after_start) = shader
        .split_once("struct Params {")
        .expect("presentation shader declares Params");
    let (body, _) = after_start
        .split_once('}')
        .expect("presentation Params declaration is closed");
    let fields = body
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_suffix(',')
                .and_then(|line| line.split_once(':'))
                .map(|(name, _)| name.trim())
        })
        .collect::<Vec<_>>();
    assert_eq!(
        fields,
        [
            "mode",
            "encode_srgb",
            "dither_bits",
            "content_mode",
            "hdr_reference_white_nits",
            "source_peak_nits",
            "output_peak_nits",
            "exposure_stops",
            "viewport_width",
            "viewport_height",
            "image_width",
            "image_height",
            "ui_white_nits",
            "view_center_x",
            "view_center_y",
            "view_scale",
            "background_mode",
            "center_ui",
            "source_dynamic_range",
            "_padding_0",
            "tiled_image",
            "tile_size",
            "tile_columns",
            "tile_gutter",
            "resampled_viewport",
            "_padding_1",
            "_padding_2",
            "_padding_3",
        ]
    );
}
