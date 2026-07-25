// Signal values from the HLG narrow-range pattern in ITU-R BT.2111-3,
// Table 2. Geometry is expressed proportionally so the pattern fills the
// available viewport rather than requiring a particular raster size.

const HLG_100_BARS = array<vec3f, 7>(
    vec3f(940.0, 940.0, 940.0),
    vec3f(940.0, 940.0, 64.0),
    vec3f(64.0, 940.0, 940.0),
    vec3f(64.0, 940.0, 64.0),
    vec3f(940.0, 64.0, 940.0),
    vec3f(940.0, 64.0, 64.0),
    vec3f(64.0, 64.0, 940.0),
);

const HLG_75_BARS = array<vec3f, 7>(
    vec3f(721.0, 721.0, 721.0),
    vec3f(721.0, 721.0, 64.0),
    vec3f(64.0, 721.0, 721.0),
    vec3f(64.0, 721.0, 64.0),
    vec3f(721.0, 64.0, 721.0),
    vec3f(721.0, 64.0, 64.0),
    vec3f(64.0, 64.0, 721.0),
);

const HLG_BAR_END_X = array<f32, 7>(
    446.0, 652.0, 858.0, 1062.0, 1268.0, 1474.0, 1680.0,
);

const HLG_STAIR = array<f32, 13>(
    4.0, 64.0, 152.0, 239.0, 327.0, 414.0, 502.0,
    590.0, 677.0, 765.0, 852.0, 940.0, 1019.0,
);

// Right edges in the 1920-pixel reference raster. The -7% step occupies the
// complete first colour bar. Subsequent steps occupy d/2 or e/2, so every pair
// lines up with the yellow-through-blue colour-bar boundaries in Figure 1.
const HLG_STAIR_END_X = array<f32, 13>(
    446.0, 549.0, 652.0, 755.0, 858.0, 960.0, 1062.0,
    1165.0, 1268.0, 1371.0, 1474.0, 1577.0, 1680.0,
);

fn main_bar_code(uv: vec2f, use_100_percent: bool) -> vec3f {
    let reference_x = clamp(uv.x, 0.0, 0.999999) * 1920.0;
    if reference_x < 240.0 || reference_x >= 1680.0 { return vec3f(414.0); }
    for (var index = 0u; index < 7u; index++) {
        if reference_x < HLG_BAR_END_X[index] {
            return select(HLG_75_BARS[index], HLG_100_BARS[index], use_100_percent);
        }
    }
    return vec3f(414.0);
}

fn stair_code(x: f32) -> vec3f {
    let reference_x = clamp(x, 0.0, 0.999999) * 1920.0;
    if reference_x < 240.0 || reference_x >= 1680.0 { return vec3f(721.0); }
    for (var index = 0u; index < 13u; index++) {
        if reference_x < HLG_STAIR_END_X[index] {
            return vec3f(HLG_STAIR[index]);
        }
    }
    return vec3f(721.0);
}

fn ramp_code(x: f32) -> vec3f {
    let reference_x = clamp(x, 0.0, 0.999999) * 1920.0;

    // The first c = 240 reference pixels are 0% black.
    if reference_x < 240.0 { return vec3f(64.0); }

    let ramp_x = reference_x - 240.0;

    // Table 5 divides the A = 1680-pixel ramp into B + C + D. B holds the
    // minimum permitted narrow-range code.
    if ramp_x < 559.0 { return vec3f(4.0); }

    // C = 1014 pixels carries one primary 10-bit code per reference pixel,
    // covering codes 5 through 1018 inclusive.
    if ramp_x < 1573.0 {
        return vec3f(5.0 + floor(ramp_x - 559.0));
    }

    // D = 107 pixels holds the maximum permitted narrow-range code.
    return vec3f(1019.0);
}

fn lower_code(x: f32) -> vec3f {
    // Reference positions use the 1920-pixel proportions from Figure 1, but
    // are evaluated in normalized coordinates at any viewport size.
    let reference_x = clamp(x, 0.0, 0.999999) * 1920.0;
    if reference_x < 80.0 { return vec3f(713.0, 719.0, 316.0); }
    if reference_x < 160.0 { return vec3f(538.0, 709.0, 718.0); }
    if reference_x < 240.0 { return vec3f(512.0, 706.0, 296.0); }
    if reference_x < 376.0 { return vec3f(64.0); }
    if reference_x < 446.0 { return vec3f(48.0); }
    if reference_x < 514.0 { return vec3f(64.0); }
    if reference_x < 584.0 { return vec3f(80.0); }
    if reference_x < 652.0 { return vec3f(64.0); }
    if reference_x < 722.0 { return vec3f(99.0); }
    if reference_x < 960.0 { return vec3f(64.0); }
    if reference_x < 1398.0 { return vec3f(721.0); }
    if reference_x < 1680.0 { return vec3f(64.0); }
    if reference_x < 1760.0 { return vec3f(651.0, 286.0, 705.0); }
    if reference_x < 1840.0 { return vec3f(639.0, 269.0, 164.0); }
    return vec3f(227.0, 147.0, 702.0);
}

fn bt2111_hlg_signal(uv: vec2f) -> vec3f {
    var code = vec3f(0.0);
    if uv.y < 1.0 / 12.0 {
        code = main_bar_code(uv, true);
    } else if uv.y < 7.0 / 12.0 {
        code = main_bar_code(uv, false);
    } else if uv.y < 8.0 / 12.0 {
        code = stair_code(uv.x);
    } else if uv.y < 9.0 / 12.0 {
        code = ramp_code(uv.x);
    } else {
        code = lower_code(uv.x);
    }
    // Expand 10-bit narrow-range RGB codes into the encoded HLG signal that a
    // decoder supplies before applying the inverse OETF and source OOTF.
    return (code - vec3f(64.0)) / 876.0;
}
