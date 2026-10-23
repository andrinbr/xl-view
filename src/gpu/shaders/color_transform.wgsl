fn finite_or_zero(value: f32) -> f32 {
    // WGSL has no portable isnan/isinf built-ins.
    if value != value || abs(value) > 3.402823e38 { return 0.0; }
    return value;
}

fn sanitize(value: f32) -> f32 {
    return max(finite_or_zero(value), 0.0);
}

const PQ_PEAK_NITS: f32 = 10000.0;
const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;
const ICTCP_GAMUT_SEARCH_ITERATIONS: u32 = 12u;
// Clip colors at most 0.1% outside BT.709 (relative to the largest channel)
// instead of gamut mapping them. This covers f16 and conversion rounding for
// every hue. Clipping such small excursions stays within about one
// just-noticeable difference of the source.
const GAMUT_BOUNDARY_TOLERANCE: f32 = 1e-3;

// SMPTE ST 2084 inverse EOTF, with luminance normalized to 10,000 nits.
fn pq_oetf_scalar(normalized_luminance: f32) -> f32 {
    let powered = pow(max(normalized_luminance, 0.0), PQ_M1);
    return pow(
        (PQ_C1 + PQ_C2 * powered) / (1.0 + PQ_C3 * powered),
        PQ_M2,
    );
}

// SMPTE ST 2084 EOTF, returning luminance normalized to 10,000 nits.
fn pq_eotf_scalar(encoded: f32) -> f32 {
    let powered = pow(max(encoded, 0.0), 1.0 / PQ_M2);
    let numerator = max(powered - PQ_C1, 0.0);
    let denominator = max(PQ_C2 - PQ_C3 * powered, 1.175494e-38);
    return pow(numerator / denominator, 1.0 / PQ_M1);
}

fn pq_oetf(normalized_luminance: vec3f) -> vec3f {
    return vec3f(
        pq_oetf_scalar(normalized_luminance.x),
        pq_oetf_scalar(normalized_luminance.y),
        pq_oetf_scalar(normalized_luminance.z),
    );
}

fn pq_eotf(encoded: vec3f) -> vec3f {
    return vec3f(
        pq_eotf_scalar(encoded.x),
        pq_eotf_scalar(encoded.y),
        pq_eotf_scalar(encoded.z),
    );
}

// Only the BT.2111 diagnostics pattern uses this. Clamping shows its -7%
// sub-black step as black. Decoded HLG images use the CPU
// color::hlg_inverse_oetf, which mirrors BT.2100's extended range below zero.
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
    let peak_nits = max(display_peak_nits, 1.0);
    if peak_nits >= 400.0 && peak_nits <= 2000.0 {
        return 1.2 + 0.42 * log2(peak_nits / 1000.0) / log2(10.0);
    }
    return 1.2 * pow(1.111, log2(peak_nits / 1000.0));
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

// Recommendation ITU-R BT.2100 PQ-based ICtCp matrices. LMS_PRIME is
// PQ-encoded LMS with absolute luminance normalized to 10,000 cd/m².
const BT2020_TO_LMS = mat3x3f(
    vec3f(1688.0, 683.0, 99.0) / 4096.0,
    vec3f(2146.0, 2951.0, 309.0) / 4096.0,
    vec3f(262.0, 462.0, 3688.0) / 4096.0,
);
const LMS_TO_BT2020 = mat3x3f(
    vec3f(3.4366067, -0.7913296, -0.0259499),
    vec3f(-2.5064521, 1.9836005, -0.0989137),
    vec3f(0.0698454, -0.1922709, 1.1248636),
);
const LMS_PRIME_TO_ICTCP = mat3x3f(
    vec3f(2048.0, 6610.0, 17933.0) / 4096.0,
    vec3f(2048.0, -13613.0, -17390.0) / 4096.0,
    vec3f(0.0, 7003.0, -543.0) / 4096.0,
);
const ICTCP_TO_LMS_PRIME = mat3x3f(
    vec3f(1.0, 1.0, 1.0),
    vec3f(0.0086090, -0.0086090, 0.5600313),
    vec3f(0.1110296, -0.1110296, -0.3206272),
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

fn tone_map_pq_hermite(
    value: f32,
    source_peak_nits: f32,
    display_peak_nits: f32,
    source_peak_pq: f32,
    output_peak_pq: f32,
) -> f32 {
    if source_peak_nits <= display_peak_nits { return value; }
    if value >= source_peak_nits { return display_peak_nits; }

    let input = pq_oetf_scalar(value / PQ_PEAK_NITS) / source_peak_pq;
    let output_peak = output_peak_pq / source_peak_pq;
    let knee = max(1.5 * output_peak - 0.5, 0.0);
    if input <= knee { return value; }

    // Report ITU-R BT.2408-8 Annex 5 PQ-domain Hermite EETF, assuming
    // zero-valued mastering and target black endpoints (LB = Lmin = 0), and
    // applied with the Annex's maxRGB option below. When an extreme peak ratio
    // clamps the knee to black, reduce its tangent too so the extension stays
    // monotonic and concave.
    let knee_slope = min(1.0, 3.0 * output_peak);
    let position = clamp((input - knee) / (1.0 - knee), 0.0, 1.0);
    let position2 = position * position;
    let position3 = position2 * position;
    let mapped = (2.0 * position3 - 3.0 * position2 + 1.0) * knee
        + (position3 - 2.0 * position2 + position) * (1.0 - knee) * knee_slope
        + (-2.0 * position3 + 3.0 * position2) * output_peak;

    return clamp(
        pq_eotf_scalar(mapped * source_peak_pq) * PQ_PEAK_NITS,
        0.0,
        display_peak_nits,
    );
}

fn tone_map_bt2020_max_rgb(
    rgb_nits: vec3f,
    source_peak_nits: f32,
    display_peak_nits: f32,
    source_peak_pq: f32,
    output_peak_pq: f32,
) -> vec3f {
    if source_peak_nits <= display_peak_nits { return rgb_nits; }

    let maximum = max(rgb_nits.r, max(rgb_nits.g, rgb_nits.b));
    if maximum <= 0.0 { return rgb_nits; }

    let mapped_maximum = tone_map_pq_hermite(
        maximum,
        source_peak_nits,
        display_peak_nits,
        source_peak_pq,
        output_peak_pq,
    );
    return rgb_nits * (mapped_maximum / maximum);
}

fn bt2020_nits_to_ictcp(rgb: vec3f) -> vec3f {
    let lms_prime = pq_oetf(max(BT2020_TO_LMS * rgb, vec3f(0.0)) / PQ_PEAK_NITS);
    return LMS_PRIME_TO_ICTCP * lms_prime;
}

fn ictcp_to_bt2020_nits(ictcp: vec3f) -> vec3f {
    let lms = pq_eotf(ICTCP_TO_LMS_PRIME * ictcp) * PQ_PEAK_NITS;
    return LMS_TO_BT2020 * lms;
}

fn rgb_is_within_cube(rgb: vec3f, upper: f32) -> bool {
    // Comparisons also reject NaN and infinities without extra built-ins.
    return all(rgb >= vec3f(0.0)) && all(rgb <= vec3f(upper));
}

// Numerical safety projection for values already expressed in the output RGB
// space: move toward the equal-luminance neutral until the RGB cube is reached.
fn bound_rgb_preserving_luminance(rgb: vec3f, luminance: f32, upper: f32) -> vec3f {
    let neutral = clamp(luminance, 0.0, upper);
    let chroma = rgb - vec3f(neutral);
    var scale = 1.0;
    for (var channel = 0u; channel < 3u; channel++) {
        if chroma[channel] < 0.0 {
            scale = min(scale, neutral / -chroma[channel]);
        } else if chroma[channel] > 0.0 {
            scale = min(scale, (upper - neutral) / chroma[channel]);
        }
    }
    return clamp(vec3f(neutral) + chroma * clamp(scale, 0.0, 1.0), vec3f(0.0), vec3f(upper));
}

// Report ITU-R BT.2446-1 § 5.2.4 hard destination-volume reduction:
// preserve ICtCp intensity and hue while desaturating to the BT.709 surface.
// This lean hard mapper sends every color beyond the boundary on a
// shared constant-I/hue ray to the same surface point, so saturated gradients
// can flatten.
fn gamut_map_bt2020_to_bt709_ictcp_hard(rgb_nits: vec3f, upper: f32) -> vec3f {
    var finite_rgb_nits = vec3f(
        finite_or_zero(rgb_nits.r), finite_or_zero(rgb_nits.g), finite_or_zero(rgb_nits.b),
    );
    let direct = BT2020_TO_BT709 * finite_rgb_nits;
    let tolerance = max(max(abs(direct.r), abs(direct.g)), abs(direct.b))
        * GAMUT_BOUNDARY_TOLERANCE;
    if all(direct >= vec3f(-tolerance)) && all(direct <= vec3f(upper + tolerance)) {
        return clamp(direct, vec3f(0.0), vec3f(upper));
    }

    if any(finite_rgb_nits < vec3f(0.0)) {
        finite_rgb_nits = bound_rgb_preserving_luminance(
            finite_rgb_nits, luma_bt2020(finite_rgb_nits), upper,
        );
    }
    let ictcp = bt2020_nits_to_ictcp(finite_rgb_nits);
    let neutral_bt709 = vec3f(pq_eotf_scalar(ictcp.x) * PQ_PEAK_NITS);
    if !rgb_is_within_cube(neutral_bt709, upper) {
        return bound_rgb_preserving_luminance(direct, luma_bt709(direct), upper);
    }

    var inside = 0.0;
    var outside = 1.0;
    var inside_bt709 = neutral_bt709;
    // Bisection assumes one monotone in/out crossing along this
    // constant-I ICtCp chroma ray. PQ makes the corresponding RGB path
    // nonlinear, so that is not guaranteed globally. Keeping only confirmed
    // in-cube samples makes this safe.
    //
    // Each out-of-gamut fragment runs all 12 inverse ICtCp steps:
    // 72 scalar pow evaluations in this loop, plus the neutral conversion.
    // Mixed subgroups may remain occupied until their out-of-gamut lanes finish.
    for (
        var iteration = 0u;
        iteration < ICTCP_GAMUT_SEARCH_ITERATIONS;
        iteration++
    ) {
        let scale = (inside + outside) * 0.5;
        let candidate = vec3f(ictcp.x, ictcp.yz * scale);
        let bt709 = BT2020_TO_BT709 * ictcp_to_bt2020_nits(candidate);
        if rgb_is_within_cube(bt709, upper) {
            inside = scale;
            inside_bt709 = bt709;
        } else {
            outside = scale;
        }
    }

    return clamp(inside_bt709, vec3f(0.0), vec3f(upper));
}
