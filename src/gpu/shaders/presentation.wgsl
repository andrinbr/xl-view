struct Params {
    // 0 = SDR sRGB, 1 = extended-linear scRGB, 2 = HDR10 PQ, 3 = HLG.
    mode: u32,
    // Non-sRGB SDR attachments need the shader to apply the sRGB OETF.
    encode_srgb: u32,
    // Zero for floating point; otherwise the destination UNORM bit depth.
    dither_bits: u32,
    // 0 = empty state, 1 = diagnostics pattern, 2 = decoded image.
    content_mode: u32,
    hdr_reference_white_nits: f32,
    source_peak_nits: f32,
    output_peak_nits: f32,
    exposure_stops: f32,
    viewport_width: u32,
    viewport_height: u32,
    image_width: u32,
    image_height: u32,
    ui_white_nits: f32,
    view_center_x: f32,
    view_center_y: f32,
    view_scale: f32,
    // 0 = black, 1 = checkerboard, 2 = white, 3 = 18% middle gray.
    background_mode: u32,
    center_ui: u32,
    // 0 = SDR, 1 = PQ, 2 = HLG.
    source_dynamic_range: u32,
    _padding_0: u32,
    tiled_image: u32,
    tile_size: u32,
    tile_columns: u32,
    tile_gutter: u32,
    resampled_viewport: u32,
    _padding_1: u32,
    _padding_2: u32,
    _padding_3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var coarse_texture: texture_2d<f32>;
@group(0) @binding(2) var image_sampler: sampler;
// Premultiplied, linear-sRGB SDR UI intermediate.
@group(0) @binding(3) var ui_texture: texture_2d<f32>;
@group(0) @binding(4) var tile_texture: texture_2d_array<f32>;
@group(0) @binding(5) var<storage, read> tile_slots: array<u32>;
@group(0) @binding(6) var resampled_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    return VertexOutput(vec4f(x, y, 0.0, 1.0), vec2f(x, -y) * 0.5 + 0.5);
}

fn background(position: vec2u) -> vec3f {
    if params.background_mode == 0u { return vec3f(0.0); }
    if params.background_mode == 2u { return vec3f(1.0); }
    if params.background_mode == 3u { return vec3f(0.18); }
    let alternate = ((position.x / 16u) + (position.y / 16u)) & 1u;
    // Neutral values are already linear and relative to HDR reference white.
    return select(vec3f(0.18), vec3f(0.32), alternate == 1u);
}

fn canonical_input(uv: vec2f, position: vec2u) -> vec4f {
    if params.content_mode == 0u {
        return vec4f(vec3f(0.012), 1.0);
    }
    if params.content_mode == 1u {
        return vec4f(hlg_input_to_canonical(
            bt2111_hlg_signal(uv),
            params.source_peak_nits,
            params.hdr_reference_white_nits,
        ), 1.0);
    }

    let viewport_center = vec2f(f32(params.viewport_width), f32(params.viewport_height)) * 0.5;
    let image_position = vec2f(params.view_center_x, params.view_center_y)
        + (vec2f(position) + vec2f(0.5) - viewport_center) / params.view_scale;
    // Derive LOD from continuous image coordinates. Tile-local UV wraps at
    // every boundary; implicit derivatives there see an almost-one-texture
    // jump and incorrectly select the smallest mip for the boundary quad.
    let image_position_dx = dpdx(image_position);
    let image_position_dy = dpdy(image_position);
    let image_extent = vec2f(f32(params.image_width), f32(params.image_height));
    let image_uv = image_position / image_extent;

    var rgba = vec4f(0.0);
    if params.resampled_viewport == 1u {
        // The resampled texture has one texel per current physical viewport
        // pixel, so no second reconstruction filter is needed here.
        rgba = textureLoad(resampled_texture, vec2i(position), 0);
    } else {
        if any(image_uv < vec2f(0.0)) || any(image_uv > vec2f(1.0)) {
            return vec4f(0.0);
        }
        rgba = textureSampleGrad(
            coarse_texture,
            image_sampler,
            image_uv,
            image_position_dx / image_extent,
            image_position_dy / image_extent,
        );
        if params.tiled_image == 1u {
            let maximum_position =
                vec2f(f32(params.image_width), f32(params.image_height)) - vec2f(0.0001);
            let bounded_position = clamp(image_position, vec2f(0.0), maximum_position);
            let tile_position = vec2u(bounded_position) / params.tile_size;
            let logical_tile = tile_position.y * params.tile_columns + tile_position.x;
            let slot = tile_slots[logical_tile];
            if slot != 0xffffffffu {
                let tile_origin = vec2f(tile_position * params.tile_size);
                let tile_extent = f32(params.tile_size + params.tile_gutter * 2u);
                let tile_uv =
                    (bounded_position - tile_origin + vec2f(f32(params.tile_gutter))) / tile_extent;
                rgba = textureSampleGrad(
                    tile_texture,
                    image_sampler,
                    tile_uv,
                    i32(slot),
                    image_position_dx / tile_extent,
                    image_position_dy / tile_extent,
                );
            }
        }
    }
    let alpha = clamp(sanitize(rgba.a), 0.0, 1.0);
    // The sampled texture is premultiplied for fringe-free linear filtering;
    // both operands are canonical linear BT.2020 values. Composite before
    // exposure, tone mapping, gamut mapping, and transfer encoding.
    return vec4f(
        sanitize(rgba.r), sanitize(rgba.g), sanitize(rgba.b),
        alpha,
    );
}

fn ui_input(position: vec2u) -> vec4f {
    let size = textureDimensions(ui_texture);
    var origin = vec2u(0u);
    if params.center_ui == 1u {
        let viewport = vec2u(params.viewport_width, params.viewport_height);
        origin = (viewport - min(viewport, size)) / 2u;
    }
    if any(position < origin) { return vec4f(0.0); }
    let local_position = position - origin;
    if any(local_position >= size) { return vec4f(0.0); }
    return textureLoad(ui_texture, vec2i(local_position), 0);
}

fn output_transform(canonical: vec4f, ui: vec4f, position: vec2u) -> vec3f {
    let exposure = exp2(params.exposure_stops);
    let image = vec3f(
        sanitize(canonical.r), sanitize(canonical.g), sanitize(canonical.b),
    ) * exposure;
    let image_alpha = clamp(sanitize(canonical.a), 0.0, 1.0);
    let working = image + background(position) * (1.0 - image_alpha);
    let ui_alpha = clamp(sanitize(ui.a), 0.0, 1.0);
    let ui_linear_bt709 = vec3f(sanitize(ui.r), sanitize(ui.g), sanitize(ui.b));

    switch params.mode {
        case 1u: {
            // scRGB deliberately bypasses tone mapping and positive clamping.
            let image_nits = gamut_map(
                BT2020_TO_BT709 * working * params.hdr_reference_white_nits,
                luma_bt709(BT2020_TO_BT709 * working * params.hdr_reference_white_nits),
                -1.0,
            );
            return (image_nits * (1.0 - ui_alpha)
                + ui_linear_bt709 * params.ui_white_nits) / 80.0;
        }
        case 2u: {
            let image_nits = bounded_bt2020_nits(
                working,
                params.hdr_reference_white_nits,
                params.output_peak_nits,
            );
            let ui_nits = BT709_TO_BT2020 * (ui_linear_bt709 * params.ui_white_nits);
            let display_nits = image_nits * (1.0 - ui_alpha) + ui_nits;
            return dither_encoded(
                pq_oetf(display_nits / 10000.0),
                position,
                params.dither_bits,
            );
        }
        case 3u: {
            let image_nits = bounded_bt2020_nits(
                working,
                params.hdr_reference_white_nits,
                params.output_peak_nits,
            );
            let ui_nits = BT709_TO_BT2020 * (ui_linear_bt709 * params.ui_white_nits);
            let display = (image_nits * (1.0 - ui_alpha) + ui_nits)
                / params.output_peak_nits;
            let display_luminance = clamp(luma_bt2020(display), 0.0, 1.0);
            var scene = vec3f(0.0);
            if display_luminance > 0.0 {
                let gamma = hlg_system_gamma(params.output_peak_nits);
                scene = display / pow(display_luminance, (gamma - 1.0) / gamma);
            }
            return dither_encoded(hlg_oetf(scene), position, params.dither_bits);
        }
        default: {
            if params.source_dynamic_range == 0u {
                let image_bt709 = BT2020_TO_BT709 * working;
                let image_relative = gamut_map(
                    image_bt709,
                    luma_bt709(image_bt709),
                    1.0,
                );
                let linear = clamp(
                    image_relative * (1.0 - ui_alpha) + ui_linear_bt709,
                    vec3f(0.0), vec3f(1.0),
                );
                let encoded = dither_encoded(srgb_oetf(linear), position, params.dither_bits);
                if params.encode_srgb == 1u { return encoded; }
                return srgb_eotf(encoded);
            }
            // Keep metadata fixed so exposure moves pixels through the curve
            // instead of renormalizing its input range.
            let bt709 = BT2020_TO_BT709 * mapped_bt2020_nits(
                working,
                params.hdr_reference_white_nits,
                params.source_peak_nits,
                params.output_peak_nits,
                params.source_dynamic_range,
            );
            // Resolve gamut once in destination primaries; channel clipping
            // avoids adding neutral components to saturated highlights.
            let image_nits = clamp(
                bt709,
                vec3f(0.0),
                vec3f(params.output_peak_nits),
            );
            let linear = clamp(
                image_nits * (1.0 - ui_alpha) + ui_linear_bt709 * params.ui_white_nits,
                vec3f(0.0), vec3f(params.output_peak_nits),
            ) / params.output_peak_nits;
            let encoded = dither_encoded(srgb_oetf(linear), position, params.dither_bits);
            if params.encode_srgb == 1u { return encoded; }
            // The attachment applies the OETF; return an encoded-domain dither
            // converted back to linear so quantization still receives it once.
            return srgb_eotf(encoded);
        }
    }
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let position = vec2u(input.position.xy);
    let ui = ui_input(position);
    return vec4f(output_transform(canonical_input(input.uv, position), ui, position), 1.0);
}
