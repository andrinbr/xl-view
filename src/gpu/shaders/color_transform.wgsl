fn sanitize(value: f32) -> f32 {
    // WGSL has no portable isnan/isinf built-ins. NaN is unequal to itself;
    // positive infinity exceeds the largest finite f32, and negative infinity
    // is covered by the negative-input policy.
    if value != value || value > 3.402823e38 || value < 0.0 { return 0.0; }
    return value;
}

fn srgb_oetf(linear: vec3f) -> vec3f {
    let lo = linear * 12.92;
    let hi = 1.055 * pow(max(linear, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, linear <= vec3f(0.0031308));
}

fn srgb_eotf(encoded: vec3f) -> vec3f {
    let lo = encoded / 12.92;
    let hi = pow((max(encoded, vec3f(0.0)) + 0.055) / 1.055, vec3f(2.4));
    return select(hi, lo, encoded <= vec3f(0.04045));
}

// SMPTE ST 2084 inverse EOTF, with luminance normalized to 10,000 nits.
fn pq_oetf(normalized_luminance: vec3f) -> vec3f {
    let m1 = 0.1593017578125;
    let m2 = 78.84375;
    let c1 = 0.8359375;
    let c2 = 18.8515625;
    let c3 = 18.6875;
    let powered = pow(max(normalized_luminance, vec3f(0.0)), vec3f(m1));
    return pow((c1 + c2 * powered) / (1.0 + c3 * powered), vec3f(m2));
}

fn hlg_oetf(scene_linear: vec3f) -> vec3f {
    let a = 0.17883277;
    let b = 0.28466892;
    let c = 0.55991073;
    let lo = sqrt(3.0 * max(scene_linear, vec3f(0.0)));
    let hi = a * log(12.0 * max(scene_linear, vec3f(1.0 / 12.0)) - b) + c;
    return select(hi, lo, scene_linear <= vec3f(1.0 / 12.0));
}

fn hlg_inverse_oetf(encoded: vec3f) -> vec3f {
    let value = max(encoded, vec3f(0.0));
    let a = 0.17883277;
    let b = 0.28466892;
    let c = 0.55991073;
    let lo = value * value / 3.0;
    let hi = (exp((value - c) / a) + b) / 12.0;
    return select(hi, lo, value <= vec3f(0.5));
}

fn hlg_system_gamma(display_peak_nits: f32) -> f32 {
    return 1.2 + 0.42 * log2(max(display_peak_nits, 1.0) / 1000.0) / log2(10.0);
}

// Matrices are column-major in WGSL.
const BT2020_TO_BT709 = mat3x3f(
    vec3f(1.6604910, -0.1245505, -0.0181508),
    vec3f(-0.5876411, 1.1328999, -0.1005789),
    vec3f(-0.0728499, -0.0083494, 1.1187297),
);
const BT709_TO_BT2020 = mat3x3f(
    vec3f(0.6274039, 0.0690973, 0.0163914),
    vec3f(0.3292830, 0.9195404, 0.0880133),
    vec3f(0.0433131, 0.0113623, 0.8955953),
);

fn luma_bt2020(rgb: vec3f) -> f32 {
    return dot(rgb, vec3f(0.2627, 0.6780, 0.0593));
}

fn luma_bt709(rgb: vec3f) -> f32 {
    return dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
}

fn hlg_input_to_canonical(
    encoded: vec3f,
    source_peak_nits: f32,
    hdr_reference_white_nits: f32,
) -> vec3f {
    let scene = hlg_inverse_oetf(encoded);
    let luminance = luma_bt2020(scene);
    if luminance <= 0.0 { return vec3f(0.0); }
    let gamma = hlg_system_gamma(source_peak_nits);
    let ootf = pow(luminance, gamma - 1.0);
    return scene * ootf * (source_peak_nits / hdr_reference_white_nits);
}

fn tone_map_luminance(
    luminance: f32,
    hdr_reference_white_nits: f32,
    source_peak_nits: f32,
    display_peak_nits: f32,
) -> f32 {
    let white = min(hdr_reference_white_nits, display_peak_nits * 0.75);
    if luminance <= white || source_peak_nits <= display_peak_nits {
        return min(luminance, display_peak_nits);
    }
    let headroom = max(display_peak_nits - white, 1e-20);
    let highlight = luminance - white;
    return white + highlight / (1.0 + highlight / headroom);
}

// Hue-preserving destination-gamut mapping along the line from equal-luminance
// neutral to the input color. A negative upper bound means no positive clamp.
fn gamut_map(rgb: vec3f, luminance: f32, upper: f32) -> vec3f {
    let maximum = select(3.402823e38, upper, upper >= 0.0);
    let neutral = clamp(luminance, 0.0, maximum);
    let chroma = rgb - vec3f(neutral);
    var scale = 1.0;
    for (var channel = 0u; channel < 3u; channel++) {
        if chroma[channel] < 0.0 {
            scale = min(scale, neutral / -chroma[channel]);
        } else if chroma[channel] > 0.0 && upper >= 0.0 {
            scale = min(scale, (maximum - neutral) / chroma[channel]);
        }
    }
    return clamp(vec3f(neutral) + chroma * clamp(scale, 0.0, 1.0), vec3f(0.0), vec3f(maximum));
}

fn bounded_bt2020_nits(
    working: vec3f,
    hdr_reference_white_nits: f32,
    encoding_peak_nits: f32,
) -> vec3f {
    let nits = working * hdr_reference_white_nits;
    return gamut_map(nits, luma_bt2020(nits), encoding_peak_nits);
}

fn mapped_bt2020_nits(
    working: vec3f,
    hdr_reference_white_nits: f32,
    source_peak_nits: f32,
    display_peak_nits: f32,
    source_dynamic_range: u32,
) -> vec3f {
    let nits = working * hdr_reference_white_nits;
    if source_dynamic_range == 0u {
        return gamut_map(nits, luma_bt2020(nits), display_peak_nits);
    }
    let luminance = luma_bt2020(nits);
    var scaled = vec3f(0.0);
    if luminance > 0.0 {
        scaled = nits * (tone_map_luminance(
            luminance,
            hdr_reference_white_nits,
            source_peak_nits,
            display_peak_nits,
        ) / luminance);
    }
    return gamut_map(scaled, luma_bt2020(scaled), display_peak_nits);
}
