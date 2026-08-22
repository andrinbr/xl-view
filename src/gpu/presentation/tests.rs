use super::*;

fn distinct_parameters() -> PresentationParams {
    PresentationParams {
        pq_output: true,
        content_mode: 4,
        background_mode: 17,
        hdr_reference_white_nits: 5.0,
        source_peak_nits: 6.0,
        output_peak_nits: 7.0,
        exposure_stops: 8.0,
        effective_source_peak_pq: 9.0,
        output_peak_pq: 10.0,
        viewport_width: 11,
        viewport_height: 12,
        image_width: 13,
        image_height: 14,
        view_center_x: 15.0,
        view_center_y: 16.0,
        view_scale: 17.0,
        center_ui: true,
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
            1,                  // 0: pq_output
            4,                  // 1: content_mode
            17,                 // 2: background_mode
            5.0_f32.to_bits(),  // 3: hdr_reference_white_nits
            6.0_f32.to_bits(),  // 4: source_peak_nits
            7.0_f32.to_bits(),  // 5: output_peak_nits
            8.0_f32.to_bits(),  // 6: exposure_stops
            9.0_f32.to_bits(),  // 7: effective_source_peak_pq
            10.0_f32.to_bits(), // 8: output_peak_pq
            11,                 // 9: viewport_width
            12,                 // 10: viewport_height
            13,                 // 11: image_width
            14,                 // 12: image_height
            15.0_f32.to_bits(), // 13: view_center_x
            16.0_f32.to_bits(), // 14: view_center_y
            17.0_f32.to_bits(), // 15: view_scale
            1,                  // 16: center_ui
            1,                  // 17: tiled_image
            22,                 // 18: tile_size
            23,                 // 19: tile_columns
            24,                 // 20: tile_gutter
            1,                  // 21: resampled_viewport
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
            "pq_output",
            "content_mode",
            "background_mode",
            "hdr_reference_white_nits",
            "source_peak_nits",
            "output_peak_nits",
            "exposure_stops",
            "effective_source_peak_pq",
            "output_peak_pq",
            "viewport_width",
            "viewport_height",
            "image_width",
            "image_height",
            "view_center_x",
            "view_center_y",
            "view_scale",
            "center_ui",
            "tiled_image",
            "tile_size",
            "tile_columns",
            "tile_gutter",
            "resampled_viewport",
        ]
    );
}
