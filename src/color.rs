//! CPU reference implementation of xl-view's display color pipeline.
//!
//! Decoded RGB is linear BT.2020/D65. A working value of `1.0` is the
//! fixed [`HDR_REFERENCE_WHITE_NITS`], while [`SourceIntensityTarget`] and
//! [`OutputPeak`] are always absolute cd/m².

/// BT.2100/BT.2408 HDR reference white used to normalize the working space.
pub const HDR_REFERENCE_WHITE_NITS: f32 = 203.0;
/// Absolute luminance of the maximum PQ signal value (SMPTE ST 2084).
pub const PQ_PEAK_NITS: f32 = 10_000.0;
const PQ_PEAK_NITS_F64: f64 = PQ_PEAK_NITS as f64;

const BT2020_LUMA: [f64; 3] = [0.2627, 0.6780, 0.0593];
const BT709_LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];
const ICTCP_GAMUT_SEARCH_ITERATIONS: u32 = 12;
// Clip colors at most 0.1% outside BT.709 (relative to the largest channel)
// instead of gamut mapping them. This covers f16 and conversion rounding for
// every hue. Clipping such small excursions stays within about one
// just-noticeable difference of the source.
const GAMUT_BOUNDARY_TOLERANCE: f64 = 1.0e-3;

// Recommendation ITU-R BT.2100 PQ-based ICtCp matrices. LMS_PRIME is
// PQ-encoded LMS with absolute luminance normalized to 10,000 cd/m².
const BT2020_TO_LMS: [[f64; 3]; 3] = [
    [1_688.0 / 4_096.0, 2_146.0 / 4_096.0, 262.0 / 4_096.0],
    [683.0 / 4_096.0, 2_951.0 / 4_096.0, 462.0 / 4_096.0],
    [99.0 / 4_096.0, 309.0 / 4_096.0, 3_688.0 / 4_096.0],
];
const LMS_TO_BT2020: [[f64; 3]; 3] = [
    [
        3.436_606_694_333_078_4,
        -2.506_452_118_656_27,
        0.069_845_424_323_191_48,
    ],
    [
        -0.791_329_555_598_928_7,
        1.983_600_451_792_290_7,
        -0.192_270_896_193_362,
    ],
    [
        -0.025_949_899_690_592_672,
        -0.098_913_714_711_726_44,
        1.124_863_614_402_319_2,
    ],
];
const LMS_PRIME_TO_ICTCP: [[f64; 3]; 3] = [
    [2_048.0 / 4_096.0, 2_048.0 / 4_096.0, 0.0],
    [6_610.0 / 4_096.0, -13_613.0 / 4_096.0, 7_003.0 / 4_096.0],
    [17_933.0 / 4_096.0, -17_390.0 / 4_096.0, -543.0 / 4_096.0],
];
const ICTCP_TO_LMS_PRIME: [[f64; 3]; 3] = [
    [1.0, 0.008_609_037_037_932_756, 0.111_029_625_003_025_96],
    [1.0, -0.008_609_037_037_932_756, -0.111_029_625_003_025_96],
    [1.0, 0.560_031_335_710_679_1, -0.320_627_174_987_318_85],
];

/// Linear BT.2020 to linear BT.709/sRGB, both D65.
const BT2020_TO_BT709: [[f64; 3]; 3] = [
    [
        1.660_491_002_108_434_7,
        -0.587_641_138_788_549_5,
        -0.072_849_863_319_885_5,
    ],
    [
        -0.124_550_474_521_590_74,
        1.132_899_897_125_959_7,
        -0.008_349_422_604_369_154,
    ],
    [
        -0.018_150_763_354_905_256,
        -0.100_578_898_008_007_39,
        1.118_729_661_362_913_7,
    ],
];

macro_rules! positive_nits_type {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name(f64);

        impl $name {
            /// Constructs a finite, positive luminance value.
            #[must_use]
            pub fn new(nits: f64) -> Option<Self> {
                (nits.is_finite() && nits > 0.0).then_some(Self(nits))
            }

            /// Returns the absolute luminance in cd/m².
            #[must_use]
            pub const fn nits(self) -> f64 {
                self.0
            }
        }
    };
}

positive_nits_type!(
    SourceIntensityTarget,
    "Source mastering/intensity target in cd/m²."
);
positive_nits_type!(OutputPeak, "Display tone-mapping target peak in cd/m².");

/// Transfer-function class of the decoded source image.
///
/// Decoded pixels are always linear BT.2020, but SDR remains relative while
/// PQ, HLG and linear pixels carry absolute luminance
/// normalized by HDR reference white.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceDynamicRange {
    Sdr,
    Linear,
    Pq,
    Hlg,
}

impl SourceIntensityTarget {
    /// Conservative fallback when source metadata is unavailable or invalid.
    pub const FALLBACK: Self = Self(1_000.0);

    /// HDR reference white, the presentation peak of SDR sources.
    pub const REFERENCE_WHITE: Self = Self(HDR_REFERENCE_WHITE_NITS as f64);

    /// Uses valid JPEG XL metadata or the documented 1,000-nit fallback.
    #[must_use]
    pub fn from_jxl_metadata(nits: f32) -> Self {
        Self::new(f64::from(nits)).unwrap_or(Self::FALLBACK)
    }
}

/// Destination output encoding. Values returned for `SdrSrgbHardware` are
/// linear; an sRGB texture view applies the OETF exactly once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputEncoding {
    Pq,
    SdrSrgbHardware,
}

/// Parameters which do not belong to source image metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputTransform {
    pub encoding: OutputEncoding,
    pub source_intensity_target: SourceIntensityTarget,
    pub output_peak: OutputPeak,
    /// Exposure in photographic stops, applied in linear light.
    pub exposure_stops: f64,
}

impl OutputTransform {
    /// Transforms one straight-alpha canonical working-space pixel.
    #[must_use]
    pub fn transform(self, rgba: [f64; 4]) -> [f64; 4] {
        let exposure = exposure_scale(self.exposure_stops);
        let working = [rgba[0], rgba[1], rgba[2]]
            .map(finite_or_zero)
            .map(|channel| channel * exposure);
        let alpha = sanitize(rgba[3]).clamp(0.0, 1.0);
        let output = match self.encoding {
            OutputEncoding::Pq => self.to_pq(working),
            OutputEncoding::SdrSrgbHardware => self.to_sdr(working),
        };
        [output[0], output[1], output[2], alpha]
    }

    fn mapped_bt2020_nits(self, working: [f64; 3]) -> [f64; 3] {
        let reference_white = f64::from(HDR_REFERENCE_WHITE_NITS);
        let nits = working.map(|channel| channel * reference_white);
        let source_peak = (self.source_intensity_target.nits()
            * exposure_scale(self.exposure_stops))
        .max(reference_white);
        let output_peak = self.output_peak.nits();
        tone_map_bt2020_max_rgb(nits, source_peak, output_peak)
    }

    fn to_pq(self, working: [f64; 3]) -> [f64; 3] {
        let mapped = self.mapped_bt2020_nits(working);
        // PQ output retains the BT.2020 destination gamut. This is only its
        // final numerical safety projection.
        bound_rgb_preserving_luminance(mapped, BT2020_LUMA, self.output_peak.nits())
            .map(|nits| pq_oetf_f64(nits / PQ_PEAK_NITS_F64))
    }

    fn to_sdr(self, working: [f64; 3]) -> [f64; 3] {
        let bt2020 = self.mapped_bt2020_nits(working);
        // SDR changes destination gamut to BT.709 and needs perceptual mapping.
        gamut_map_bt2020_to_bt709_ictcp_hard(bt2020, self.output_peak.nits())
            .map(|nits| nits / self.output_peak.nits())
    }
}

/// IEC 61966-2-1 sRGB electro-optical transfer function.
#[must_use]
pub fn srgb_eotf(encoded: f64) -> f64 {
    let encoded = encoded.max(0.0);
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

/// IEC 61966-2-1 sRGB opto-electrical transfer function.
#[must_use]
pub fn srgb_oetf(linear: f64) -> f64 {
    let linear = linear.max(0.0);
    if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

const PQ_M1: f32 = 2_610.0 / 16_384.0;
const PQ_M2: f32 = 2_523.0 / 32.0;
const PQ_C1: f32 = 3_424.0 / 4_096.0;
const PQ_C2: f32 = 2_413.0 / 128.0;
const PQ_C3: f32 = 2_392.0 / 128.0;
const HLG_A: f32 = 0.178_832_77;
const HLG_B: f32 = 0.284_668_92;
const HLG_C: f32 = 0.559_910_7;

/// SMPTE ST 2084 EOTF, returning luminance normalized to 10,000 cd/m².
#[must_use]
#[inline]
pub fn pq_eotf(encoded: f32) -> f32 {
    let power = encoded.max(0.0).powf(PQ_M2.recip());
    ((power - PQ_C1).max(0.0) / (PQ_C2 - PQ_C3 * power).max(f32::MIN_POSITIVE)).powf(PQ_M1.recip())
}

/// SMPTE ST 2084 inverse EOTF. Input is normalized to 10,000 cd/m².
#[must_use]
#[inline]
pub fn pq_oetf(normalized_luminance: f32) -> f32 {
    let power = normalized_luminance.max(0.0).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * power) / (1.0 + PQ_C3 * power)).powf(PQ_M2)
}

/// BT.2100 HLG OETF used by source-decoding reference tests.
#[must_use]
#[inline]
pub fn hlg_oetf(scene_linear: f32) -> f32 {
    let scene_linear = scene_linear.max(0.0);
    if scene_linear <= 1.0 / 12.0 {
        (3.0 * scene_linear).sqrt()
    } else {
        HLG_A * (12.0 * scene_linear - HLG_B).ln() + HLG_C
    }
}

/// BT.2100 HLG inverse OETF to scene-linear light.
#[must_use]
#[inline]
pub fn hlg_inverse_oetf(encoded: f32) -> f32 {
    // BT.2100's extended range is symmetric below zero.
    let magnitude = encoded.abs();
    let scene = if magnitude <= 0.5 {
        magnitude * magnitude / 3.0
    } else {
        (((magnitude - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    };
    scene.copysign(encoded)
}

/// BT.2100 HLG system gamma used while decoding an HLG source.
#[must_use]
#[inline]
pub fn hlg_system_gamma(display_peak_nits: f32) -> f32 {
    let peak_nits = display_peak_nits.max(1.0);
    if (400.0..=2_000.0).contains(&peak_nits) {
        1.2 + 0.42 * (peak_nits / 1_000.0).log10()
    } else {
        1.2 * 1.111_f32.powf((peak_nits / 1_000.0).log2())
    }
}

/// Applies a 3x3 row-major matrix to an RGB/XYZ vector.
#[must_use]
fn multiply(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| dot(row, vector))
}

/// Premultiplies and composites straight-alpha foreground RGB in linear light.
#[must_use]
pub fn composite_linear(foreground: [f64; 4], background: [f64; 3]) -> [f64; 3] {
    let alpha = sanitize(foreground[3]).clamp(0.0, 1.0);
    std::array::from_fn(|index| {
        finite_or_zero(foreground[index])
            .mul_add(alpha, sanitize(background[index]) * (1.0 - alpha))
    })
}

/// Adds deterministic, zero-mean triangular noise before UNORM quantization.
/// The result remains in `[0, 1]` and is suitable for 8- or 10-bit output.
///
/// # Panics
///
/// Panics unless `bits` is in `1..=16`.
#[must_use]
pub fn dither_unorm(value: f64, x: u32, y: u32, channel: u32, bits: u8) -> f64 {
    assert!((1..=16).contains(&bits), "UNORM bit depth must be 1..=16");
    let levels = f64::from((1_u32 << bits) - 1);
    let first = hash_to_unit(x, y, channel, 0);
    let second = hash_to_unit(x, y, channel, 1);
    (sanitize(value) + (first - second) / levels).clamp(0.0, 1.0)
}

fn tone_map_pq_hermite(value: f64, source_peak: f64, output_peak: f64) -> f64 {
    let value = sanitize(value);
    if source_peak <= output_peak {
        return value;
    }

    if value >= source_peak {
        return output_peak;
    }

    let source_peak_pq = pq_oetf_f64(source_peak / PQ_PEAK_NITS_F64);
    let output_peak_pq = pq_oetf_f64(output_peak / PQ_PEAK_NITS_F64);
    let input = pq_oetf_f64(value / PQ_PEAK_NITS_F64) / source_peak_pq;
    let normalized_output_peak = output_peak_pq / source_peak_pq;
    let knee = pq_hermite_knee(normalized_output_peak);
    // Preserve the exact linear-light identity. A PQ round-trip adds rounding.
    if input <= knee {
        return value;
    }
    let mapped = tone_map_normalized_pq_hermite(input, normalized_output_peak, knee);

    (pq_eotf_f64(mapped * source_peak_pq) * PQ_PEAK_NITS_F64).clamp(0.0, output_peak)
}

fn tone_map_normalized_pq_hermite(input: f64, output_peak: f64, knee: f64) -> f64 {
    if input <= knee {
        return input;
    }

    // Report ITU-R BT.2408-8 Annex 5 PQ-domain Hermite EETF, assuming
    // zero-valued mastering and target black endpoints (LB = Lmin = 0).
    // The curve is applied with the Annex's maxRGB option below.
    // If an extreme ratio clamps the knee to black, reduce its tangent too so
    // the extension remains monotonic and concave.
    let knee_slope = (3.0 * output_peak).min(1.0);
    let position = ((input - knee) / (1.0 - knee)).clamp(0.0, 1.0);
    let position2 = position * position;
    let position3 = position2 * position;
    let mapped = (2.0 * position3 - 3.0 * position2 + 1.0) * knee
        + (position3 - 2.0 * position2 + position) * (1.0 - knee) * knee_slope
        + (-2.0 * position3 + 3.0 * position2) * output_peak;
    mapped.clamp(0.0, output_peak)
}

fn pq_hermite_knee(output_peak: f64) -> f64 {
    1.5f64.mul_add(output_peak, -0.5).max(0.0)
}

fn tone_map_bt2020_max_rgb(rgb_nits: [f64; 3], source_peak: f64, output_peak: f64) -> [f64; 3] {
    if source_peak <= output_peak {
        return rgb_nits;
    }

    let maximum = rgb_nits.into_iter().fold(0.0_f64, f64::max);
    if maximum <= 0.0 {
        return rgb_nits;
    }

    let gain = tone_map_pq_hermite(maximum, source_peak, output_peak) / maximum;
    rgb_nits.map(|channel| channel * gain)
}

fn pq_oetf_f64(normalized_luminance: f64) -> f64 {
    let m1 = f64::from(PQ_M1);
    let m2 = f64::from(PQ_M2);
    let c1 = f64::from(PQ_C1);
    let c2 = f64::from(PQ_C2);
    let c3 = f64::from(PQ_C3);
    let power = normalized_luminance.max(0.0).powf(m1);
    ((c1 + c2 * power) / (1.0 + c3 * power)).powf(m2)
}

fn pq_eotf_f64(encoded: f64) -> f64 {
    let m1 = f64::from(PQ_M1);
    let m2 = f64::from(PQ_M2);
    let c1 = f64::from(PQ_C1);
    let c2 = f64::from(PQ_C2);
    let c3 = f64::from(PQ_C3);
    let power = encoded.max(0.0).powf(m2.recip());
    ((power - c1).max(0.0) / (c2 - c3 * power).max(f64::MIN_POSITIVE)).powf(m1.recip())
}

fn bt2020_nits_to_ictcp(rgb: [f64; 3]) -> [f64; 3] {
    let lms_prime = multiply(BT2020_TO_LMS, rgb)
        .map(|channel| pq_oetf_f64(channel.max(0.0) / PQ_PEAK_NITS_F64));
    multiply(LMS_PRIME_TO_ICTCP, lms_prime)
}

fn ictcp_to_bt2020_nits(ictcp: [f64; 3]) -> [f64; 3] {
    let lms =
        multiply(ICTCP_TO_LMS_PRIME, ictcp).map(|channel| pq_eotf_f64(channel) * PQ_PEAK_NITS_F64);
    multiply(LMS_TO_BT2020, lms)
}

fn rgb_is_within_cube(rgb: [f64; 3], upper: f64) -> bool {
    rgb.into_iter()
        .all(|channel| channel.is_finite() && (0.0..=upper).contains(&channel))
}

// Report ITU-R BT.2446-1 § 5.2.4 hard destination-volume reduction:
// preserve ICtCp intensity and hue while desaturating to the BT.709 surface.
// This lean hard mapper sends every color beyond the boundary on a
// shared constant-I/hue ray to the same surface point, so saturated gradients
// can flatten.
fn gamut_map_bt2020_to_bt709_ictcp_hard(rgb_nits: [f64; 3], upper: f64) -> [f64; 3] {
    let rgb_nits = rgb_nits.map(finite_or_zero);
    let direct = multiply(BT2020_TO_BT709, rgb_nits);
    let tolerance = direct
        .into_iter()
        .fold(0.0, |maximum: f64, channel| maximum.max(channel.abs()))
        * GAMUT_BOUNDARY_TOLERANCE;
    if direct
        .into_iter()
        .all(|channel| (-tolerance..=upper + tolerance).contains(&channel))
    {
        return direct.map(|channel| channel.clamp(0.0, upper));
    }

    let rgb_nits = if rgb_nits.into_iter().any(|channel| channel < 0.0) {
        bound_rgb_preserving_luminance(rgb_nits, BT2020_LUMA, upper)
    } else {
        rgb_nits
    };
    let ictcp = bt2020_nits_to_ictcp(rgb_nits);
    let neutral_bt709 = [pq_eotf_f64(ictcp[0]) * PQ_PEAK_NITS_F64; 3];
    if !rgb_is_within_cube(neutral_bt709, upper) {
        // Invalid or mis-signalled values can put even the equal-intensity
        // neutral outside the target cube, so retain the bounded fallback.
        return bound_rgb_preserving_luminance(direct, BT709_LUMA, upper);
    }

    let mut inside = 0.0_f64;
    let mut outside = 1.0_f64;
    let mut inside_bt709 = neutral_bt709;
    // Bisection assumes one monotone in/out crossing along this
    // constant-I ICtCp chroma ray. PQ makes the corresponding RGB path
    // nonlinear, so that is not guaranteed globally. Keeping only confirmed
    // in-cube samples makes this safe.
    for _ in 0..ICTCP_GAMUT_SEARCH_ITERATIONS {
        let scale = (inside + outside) * 0.5;
        let candidate = [ictcp[0], ictcp[1] * scale, ictcp[2] * scale];
        let bt709 = multiply(BT2020_TO_BT709, ictcp_to_bt2020_nits(candidate));
        if rgb_is_within_cube(bt709, upper) {
            inside = scale;
            inside_bt709 = bt709;
        } else {
            outside = scale;
        }
    }

    inside_bt709.map(|channel| channel.clamp(0.0, upper))
}

// Numerical safety projection for values already expressed in the output RGB
// space: move toward the equal-luminance neutral until the RGB cube is reached.
fn bound_rgb_preserving_luminance(rgb: [f64; 3], luma: [f64; 3], upper: f64) -> [f64; 3] {
    // Preserve finite negative matrix results here: they carry the chroma
    // direction needed for line-to-neutral gamut compression.
    let rgb = rgb.map(finite_or_zero);
    let neutral = dot(rgb, luma).clamp(0.0, upper);
    let mut scale = 1.0_f64;
    for channel in rgb {
        let chroma = channel - neutral;
        if chroma < 0.0 {
            scale = scale.min(neutral / -chroma);
        } else if chroma > 0.0 {
            scale = scale.min((upper - neutral) / chroma);
        }
    }
    rgb.map(|channel| (neutral + (channel - neutral) * scale.clamp(0.0, 1.0)).clamp(0.0, upper))
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0].mul_add(right[0], left[1].mul_add(right[1], left[2] * right[2]))
}

fn sanitize(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn exposure_scale(stops: f64) -> f64 {
    if stops.is_finite() { stops.exp2() } else { 1.0 }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

fn hash_to_unit(x: u32, y: u32, channel: u32, phase: u32) -> f64 {
    let mut value = x
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add(y.wrapping_mul(0x85eb_ca6b))
        .wrapping_add(channel.wrapping_mul(0xc2b2_ae35))
        .wrapping_add(phase.wrapping_mul(0x27d4_eb2d));
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    f64::from(value) / f64::from(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1.0e-12;

    /// Linear BT.709/sRGB to CIE XYZ (D65), with matrix rows stored in order.
    const BT709_TO_XYZ_D65: [[f64; 3]; 3] = [
        [
            0.412_390_799_265_959_5,
            0.357_584_339_383_877_96,
            0.180_480_788_401_834_3,
        ],
        [
            0.212_639_005_871_510_36,
            0.715_168_678_767_755_9,
            0.072_192_315_360_733_71,
        ],
        [
            0.019_330_818_715_591_85,
            0.119_194_779_794_625_99,
            0.950_532_152_249_660_7,
        ],
    ];

    /// CIE XYZ (D65) to linear BT.709/sRGB.
    const XYZ_D65_TO_BT709: [[f64; 3]; 3] = [
        [
            3.240_969_941_904_522_6,
            -1.537_383_177_570_094,
            -0.498_610_760_293_003_4,
        ],
        [
            -0.969_243_636_280_879_6,
            1.875_967_501_507_720_2,
            0.041_555_057_407_175_59,
        ],
        [
            0.055_630_079_696_993_66,
            -0.203_976_958_888_976_52,
            1.056_971_514_242_878_6,
        ],
    ];

    /// Linear BT.2020 to CIE XYZ (D65).
    const BT2020_TO_XYZ_D65: [[f64; 3]; 3] = [
        [
            0.636_958_048_301_291_4,
            0.144_616_903_586_208_32_f64,
            0.168_880_975_164_172_1,
        ],
        [
            0.262_700_212_011_267_1,
            0.677_998_071_518_870_8,
            0.059_301_716_469_861_96,
        ],
        [0.0, 0.028_072_693_049_087_428, 1.060_985_057_710_791],
    ];

    /// CIE XYZ (D65) to linear BT.2020.
    const XYZ_D65_TO_BT2020: [[f64; 3]; 3] = [
        [
            1.716_651_187_971_268,
            -0.355_670_783_776_392_4,
            -0.253_366_281_373_659_74,
        ],
        [
            -0.666_684_351_832_489,
            1.616_481_236_634_939_5,
            0.015_768_545_813_911_13,
        ],
        [
            0.017_639_857_445_310_783,
            -0.042_770_613_257_808_524,
            0.942_103_121_235_473_8,
        ],
    ];

    /// Linear BT.709/sRGB to linear BT.2020, both D65.
    const BT709_TO_BT2020: [[f64; 3]; 3] = [
        [
            0.627_403_895_934_699,
            0.329_283_038_377_883_7,
            0.043_313_065_687_417_23,
        ],
        [
            0.069_097_289_358_232_1,
            0.919_540_395_075_458_8,
            0.011_362_315_566_309_1,
        ],
        [
            0.016_391_438_875_150_28,
            0.088_013_307_877_225_7,
            0.895_595_253_247_624_1,
        ],
    ];

    fn assert_near(actual: impl Into<f64>, expected: impl Into<f64>, tolerance: f64) {
        let actual = actual.into();
        let expected = expected.into();
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn transform(encoding: OutputEncoding, source: f64, output: f64) -> OutputTransform {
        OutputTransform {
            encoding,
            source_intensity_target: SourceIntensityTarget::new(source).unwrap(),
            output_peak: OutputPeak::new(output).unwrap(),
            exposure_stops: 0.0,
        }
    }

    #[test]
    fn transfer_functions_hit_normative_breakpoints_and_round_trip() {
        for value in [0.0, 0.003_130_8, 0.18, 1.0] {
            assert_near(srgb_eotf(srgb_oetf(value)), value, EPSILON);
        }
        assert_near(srgb_oetf(0.003_130_8), 0.040_449_936, 1.0e-9);

        for nits in [0.0_f32, 100.0, 203.0, 1_000.0, 10_000.0] {
            let normalized = nits / 10_000.0;
            assert_near(pq_eotf(pq_oetf(normalized)), normalized, 1.0e-5);
        }
        assert_near(pq_oetf(100.0 / 10_000.0), 0.508_078_421_517_399, 5.0e-6);
        assert_near(pq_oetf(1_000.0 / 10_000.0), 0.751_827_096_247_041, 5.0e-6);
        assert_near(pq_oetf(1.0), 1.0, f64::from(f32::EPSILON));

        assert_near(hlg_inverse_oetf(0.0), 0.0, f64::from(f32::EPSILON));
        assert_near(hlg_inverse_oetf(0.5), 1.0 / 12.0, f64::from(f32::EPSILON));
    }

    #[test]
    fn matrices_preserve_d65_white_and_round_trip_each_rgb_space() {
        let d65 = [0.950_455_927_051_671_6, 1.0, 1.089_057_750_759_878_4];
        for matrix in [BT709_TO_XYZ_D65, BT2020_TO_XYZ_D65] {
            let white = multiply(matrix, [1.0; 3]);
            for index in 0..3 {
                assert_near(white[index], d65[index], 2.0e-12);
            }
        }
        for (to_xyz, from_xyz) in [
            (BT709_TO_XYZ_D65, XYZ_D65_TO_BT709),
            (BT2020_TO_XYZ_D65, XYZ_D65_TO_BT2020),
        ] {
            for primary in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
                let xyz = multiply(to_xyz, primary);
                let round_trip = multiply(from_xyz, xyz);
                for index in 0..3 {
                    assert_near(round_trip[index], primary[index], 2.0e-15);
                }
            }
        }
    }

    #[test]
    fn direct_bt709_bt2020_matrices_match_xyz_composition_and_are_inverse() {
        for primary in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let bt2020_direct = multiply(BT709_TO_BT2020, primary);
            let bt2020_via_xyz = multiply(XYZ_D65_TO_BT2020, multiply(BT709_TO_XYZ_D65, primary));
            let bt709_direct = multiply(BT2020_TO_BT709, primary);
            let bt709_via_xyz = multiply(XYZ_D65_TO_BT709, multiply(BT2020_TO_XYZ_D65, primary));

            for index in 0..3 {
                assert_near(bt2020_direct[index], bt2020_via_xyz[index], 2.0e-15);
                assert_near(bt709_direct[index], bt709_via_xyz[index], 2.0e-15);
            }

            let round_trip = multiply(BT2020_TO_BT709, bt2020_direct);
            for index in 0..3 {
                assert_near(round_trip[index], primary[index], 2.0e-15);
            }
        }

        let gray = multiply(BT2020_TO_BT709, [0.42; 3]);
        for channel in gray {
            assert_near(channel, 0.42, 2.0e-15);
        }
    }

    #[test]
    fn ictcp_round_trips_bt2020_nits_and_represents_neutral_intensity() {
        for rgb in [
            [0.0; 3],
            [203.0; 3],
            [1_000.0, 250.0, 40.0],
            [50.0, 600.0, 120.0],
        ] {
            let ictcp = bt2020_nits_to_ictcp(rgb);
            let round_trip = ictcp_to_bt2020_nits(ictcp);
            for index in 0..3 {
                assert_near(round_trip[index], rgb[index], 2.0e-6);
            }
        }

        let neutral = bt2020_nits_to_ictcp([203.0; 3]);
        assert_near(neutral[0], pq_oetf_f64(203.0 / 10_000.0), 2.0e-12);
        assert_near(neutral[1], 0.0, 2.0e-7);
        assert_near(neutral[2], 0.0, 2.0e-7);
    }

    #[test]
    fn tone_mapping_curve_is_monotonic_bounded_and_never_brightens() {
        for (source_peak, output_peak) in [
            (204.0, 203.0),
            (250.0, 203.0),
            (300.0, 203.0),
            (600.0, 203.0),
            (1_000.0, 203.0),
            (4_000.0, 203.0),
            (10_000.0, 203.0),
            (10_000.0, 1.0),
            (1_000.0, 600.0),
            (1_200.0, 1_000.0),
            (4_000.0, 1_000.0),
        ] {
            let increment = source_peak / 10_000.0;
            let mut previous_value = 0.0;
            for step in 0..=10_000 {
                let input = increment * f64::from(step);
                let mapped = tone_map_pq_hermite(input, source_peak, output_peak);
                assert!(mapped >= previous_value);
                assert!(
                    mapped <= input + 1.0e-10,
                    "tone mapping brightened {input} to {mapped} nits for {source_peak} -> {output_peak} nits"
                );
                assert!(mapped <= output_peak);
                if step > 0 {
                    let slope = (mapped - previous_value) / increment;
                    assert!(slope >= -1.0e-10);
                }
                previous_value = mapped;
            }
            assert_near(
                tone_map_pq_hermite(source_peak, source_peak, output_peak),
                output_peak,
                EPSILON,
            );
        }

        let source_peak = 1_000.0;
        let output_peak = 203.0;
        let source_peak_pq = pq_oetf_f64(source_peak / 10_000.0);
        let output_peak_pq = pq_oetf_f64(output_peak / 10_000.0);
        let knee_pq = 1.5f64.mul_add(output_peak_pq / source_peak_pq, -0.5);
        let knee = pq_eotf_f64(knee_pq * source_peak_pq) * 10_000.0;
        for input in [0.0, 20.0, 50.0, knee] {
            assert_eq!(
                tone_map_pq_hermite(input, source_peak, output_peak).to_bits(),
                input.to_bits(),
            );
        }

        assert_near(
            tone_map_pq_hermite(500.0, source_peak, output_peak),
            198.856_008_948_631_37,
            1.0e-10,
        );
    }

    #[test]
    fn tone_mapping_curve_is_concave_in_pq_domain() {
        for (source_peak, output_peak) in [
            (204.0, 203.0),
            (1_000.0, 203.0),
            (4_000.0, 203.0),
            (10_000.0, 203.0),
            (10_000.0, 1.0),
            (4_000.0, 1_000.0),
        ] {
            let source_peak_pq = pq_oetf_f64(source_peak / 10_000.0);
            let output_peak_pq = pq_oetf_f64(output_peak / 10_000.0);
            let normalized_output_peak = output_peak_pq / source_peak_pq;
            let knee = pq_hermite_knee(normalized_output_peak);
            let increment = 1.0 / 10_000.0;
            let mut previous_value = 0.0;
            let mut previous_slope = 1.0;
            for step in 0..=10_000 {
                let input = increment * f64::from(step);
                let mapped = tone_map_normalized_pq_hermite(input, normalized_output_peak, knee);
                if step > 0 {
                    let slope = (mapped - previous_value) / increment;
                    assert!(
                        slope >= -1.0e-9,
                        "negative PQ slope {slope} for {source_peak} -> {output_peak} nits at {input}"
                    );
                    assert!(
                        slope <= previous_slope + 1.0e-7,
                        "PQ slope increased from {previous_slope} to {slope} for {source_peak} -> {output_peak} nits at {input}"
                    );
                    previous_slope = slope;
                }
                previous_value = mapped;
            }
        }
    }

    #[test]
    fn tone_mapping_hermite_has_continuous_endpoint_slopes() {
        for (source_peak, output_peak) in [
            (250.0, 203.0),
            (300.0, 203.0),
            (600.0, 203.0),
            (1_000.0, 600.0),
            (4_000.0, 1_000.0),
        ] {
            let source_peak_pq = pq_oetf_f64(source_peak / 10_000.0);
            let output_peak_pq = pq_oetf_f64(output_peak / 10_000.0);
            let knee_pq = 1.5f64.mul_add(output_peak_pq / source_peak_pq, -0.5);
            let knee = pq_eotf_f64(knee_pq * source_peak_pq) * 10_000.0;
            let step = (source_peak - knee) * 1.0e-6;
            let knee_left = (tone_map_pq_hermite(knee, source_peak, output_peak)
                - tone_map_pq_hermite(knee - step, source_peak, output_peak))
                / step;
            let knee_right = (tone_map_pq_hermite(knee + step, source_peak, output_peak)
                - tone_map_pq_hermite(knee, source_peak, output_peak))
                / step;
            assert_near(knee_left, 1.0, 1.0e-6);
            assert_near(knee_right, 1.0, 2.0e-4);

            let peak_left = (tone_map_pq_hermite(source_peak, source_peak, output_peak)
                - tone_map_pq_hermite(source_peak - step, source_peak, output_peak))
                / step;
            let peak_right = (tone_map_pq_hermite(source_peak + step, source_peak, output_peak)
                - tone_map_pq_hermite(source_peak, source_peak, output_peak))
                / step;
            assert!(peak_left < 1.0e-3);
            assert_near(peak_right, 0.0, EPSILON);
        }
    }

    #[test]
    fn max_rgb_tone_mapping_preserves_channel_ratios() {
        let source = [1_000.0, 250.0, 40.0];
        let mapped = tone_map_bt2020_max_rgb(source, 1_000.0, 203.0);
        assert_near(mapped[0], 203.0, EPSILON);
        assert_near(mapped[1] / mapped[0], source[1] / source[0], EPSILON);
        assert_near(mapped[2] / mapped[0], source[2] / source[0], EPSILON);

        for primary in [
            [1_000.0, 0.0, 0.0],
            [0.0, 1_000.0, 0.0],
            [0.0, 0.0, 1_000.0],
        ] {
            let mapped = tone_map_bt2020_max_rgb(primary, 1_000.0, 203.0);
            assert_near(mapped.into_iter().fold(0.0_f64, f64::max), 203.0, EPSILON);
            assert_eq!(
                mapped.into_iter().filter(|channel| *channel == 0.0).count(),
                2,
            );
        }
    }

    #[test]
    fn bright_saturated_colors_do_not_tone_map_to_white() {
        let transform = transform(OutputEncoding::Pq, 1_000.0, 203.0);
        for source_nits in [
            [1_000.0, 0.0, 0.0],
            [0.0, 1_000.0, 0.0],
            [0.0, 0.0, 1_000.0],
            [1_000.0, 1_000.0, 0.0],
            [1_000.0, 500.0, 0.0],
        ] {
            let canonical = source_nits.map(|channel| channel / 203.0);
            let output = transform.transform([canonical[0], canonical[1], canonical[2], 1.0]);
            let mapped_nits = [
                pq_eotf_f64(output[0]) * 10_000.0,
                pq_eotf_f64(output[1]) * 10_000.0,
                pq_eotf_f64(output[2]) * 10_000.0,
            ];
            let minimum = mapped_nits.into_iter().fold(f64::INFINITY, f64::min);
            let maximum = mapped_nits.into_iter().fold(0.0_f64, f64::max);
            assert_near(maximum, 203.0, 2.0e-8);
            assert!(maximum - minimum > 100.0);
        }
    }

    #[test]
    fn only_pq_and_hardware_srgb_transforms_remain_finite_and_neutral() {
        for encoding in [OutputEncoding::Pq, OutputEncoding::SdrSrgbHardware] {
            let output = transform(encoding, 4_000.0, 1_000.0).transform([1.0, 1.0, 1.0, 0.5]);
            assert!(output.iter().all(|value| value.is_finite()));
            assert_near(output[0], output[1], 1.0e-12);
            assert_near(output[1], output[2], 1.0e-12);
            assert_near(output[3], 0.5, EPSILON);
        }
    }

    #[test]
    fn reference_white_and_pq_transfer_have_pinned_values() {
        let sdr = transform(OutputEncoding::SdrSrgbHardware, 203.0, 203.0)
            .transform([1.0, 1.0, 1.0, 1.0]);
        assert_near(sdr[0], 1.0, EPSILON);

        let pq = transform(OutputEncoding::Pq, 203.0, 600.0).transform([1.0, 1.0, 1.0, 1.0]);
        assert_near(pq[0], 0.580_688_881, 2.0e-7);
        assert_near(pq_eotf_f64(pq[0]) * 10_000.0, 203.0, 2.0e-8);
        assert!((593.5..594.5).contains(&(pq[0] * 1_023.0)));
    }

    #[test]
    fn universal_mapping_hits_declared_peaks_and_rolls_highlights() {
        let mapped = tone_map_bt2020_max_rgb([1_000.0; 3], 1_000.0, 600.0);
        for channel in mapped {
            assert_near(channel, 600.0, 2.0e-5);
        }

        let transform = transform(OutputEncoding::SdrSrgbHardware, 1_000.0, 203.0);
        let below = transform.transform([500.0 / 203.0; 4])[0];
        let peak = transform.transform([1_000.0 / 203.0; 4])[0];
        assert!(below < peak);
        assert_near(peak, 1.0, 2.0e-6);
    }

    #[test]
    fn exposed_sdr_uses_effective_source_peak_without_double_exposure() {
        let sdr_transform = OutputTransform {
            exposure_stops: 1.0,
            ..transform(OutputEncoding::SdrSrgbHardware, 203.0, 203.0)
        };
        let gray = sdr_transform.transform([0.18, 0.18, 0.18, 1.0]);
        assert_near(gray[0], 0.36, 2.0e-6);
        assert_near(gray[0] * 203.0, 73.08, 5.0e-4);

        let pq_transform = OutputTransform {
            encoding: OutputEncoding::Pq,
            ..sdr_transform
        };
        let pq_gray = pq_transform.transform([0.18, 0.18, 0.18, 1.0]);
        assert_near(pq_eotf_f64(pq_gray[0]) * 10_000.0 / 203.0, 0.36, 2.0e-6);

        let highlight = sdr_transform.transform([0.75, 0.75, 0.75, 1.0])[0];
        let white = sdr_transform.transform([1.0, 1.0, 1.0, 1.0])[0];
        assert!(gray[0] < highlight && highlight < white && white <= 1.0);
    }

    #[test]
    fn pq_safety_projection_preserves_neutrals_and_reduces_chroma() {
        assert_eq!(
            bound_rgb_preserving_luminance([80.0; 3], BT709_LUMA, 203.0).map(f64::to_bits),
            [80.0; 3].map(f64::to_bits)
        );
        let converted = multiply(BT2020_TO_BT709, [203.0, 0.0, 0.0]);
        let mapped = bound_rgb_preserving_luminance(converted, BT709_LUMA, 203.0);
        let clipped = converted.map(|channel| channel.clamp(0.0, 203.0));
        assert_ne!(mapped.map(f64::to_bits), clipped.map(f64::to_bits));
        for channel in mapped {
            assert!((0.0..=203.0).contains(&channel));
        }
        for (actual, expected) in mapped.into_iter().zip([203.0, 11.8, 23.3]) {
            assert_near(actual, expected, 0.2);
        }
    }

    #[test]
    fn hard_ictcp_gamut_mapping_preserves_in_gamut_colors_exactly() {
        let source = multiply(BT709_TO_BT2020, [40.0, 100.0, 20.0]);
        let direct = multiply(BT2020_TO_BT709, source);
        assert!(rgb_is_within_cube(direct, 203.0));
        assert_eq!(
            gamut_map_bt2020_to_bt709_ictcp_hard(source, 203.0).map(f64::to_bits),
            direct.map(f64::to_bits),
        );
    }

    #[test]
    fn hard_ictcp_gamut_mapping_falls_back_when_neutral_exceeds_target_peak() {
        let source = [500.0, 0.0, 800.0];
        let direct = multiply(BT2020_TO_BT709, source);
        let ictcp = bt2020_nits_to_ictcp(source);
        let neutral_bt709 = [pq_eotf_f64(ictcp[0]) * 10_000.0; 3];
        assert!(!rgb_is_within_cube(neutral_bt709, 203.0));

        let expected = bound_rgb_preserving_luminance(direct, BT709_LUMA, 203.0);
        for (actual, expected) in expected.into_iter().zip([199.1, 170.3, 203.0]) {
            assert_near(actual, expected, 0.2);
        }
        assert_eq!(
            gamut_map_bt2020_to_bt709_ictcp_hard(source, 203.0).map(f64::to_bits),
            expected.map(f64::to_bits),
        );
    }

    #[test]
    fn cpu_and_shader_ictcp_search_iterations_match() {
        let shader = include_str!("gpu/shaders/color_transform.wgsl");
        let expected =
            format!("const ICTCP_GAMUT_SEARCH_ITERATIONS: u32 = {ICTCP_GAMUT_SEARCH_ITERATIONS}u;");
        assert!(
            shader.contains(&expected),
            "shader must contain `{expected}` to match the CPU search count",
        );
    }

    #[test]
    fn cpu_and_shader_pq_peaks_match() {
        let shader = include_str!("gpu/shaders/color_transform.wgsl");
        let expected = format!("const PQ_PEAK_NITS: f32 = {PQ_PEAK_NITS:?};");
        assert!(
            shader.contains(&expected),
            "shader must contain `{expected}` to match the CPU PQ peak",
        );
    }

    #[test]
    fn cpu_and_shader_gamut_boundary_tolerances_match() {
        let shader = include_str!("gpu/shaders/color_transform.wgsl");
        let expected =
            format!("const GAMUT_BOUNDARY_TOLERANCE: f32 = {GAMUT_BOUNDARY_TOLERANCE:e};");
        assert!(
            shader.contains(&expected),
            "shader must contain `{expected}` to match the CPU tolerance",
        );
    }

    #[test]
    fn hard_ictcp_gamut_mapping_clips_only_rounding_sized_excursions() {
        // Without the tolerance this pure blue would drop to approx 29 nits blue
        // with approx 1 nit green. The green carries real f16 storage rounding.
        let blue = multiply(BT709_TO_BT2020, [-0.001, 0.0, 35.0]);
        let green = multiply(BT709_TO_BT2020, [0.0, 0.548_828_125, 0.0])
            .map(|channel| half::f16::from_f64(channel).to_f64() * 203.0);
        for rounding in [blue, green] {
            let direct = multiply(BT2020_TO_BT709, rounding);
            assert!(!rgb_is_within_cube(direct, 203.0));
            assert_eq!(
                gamut_map_bt2020_to_bt709_ictcp_hard(rounding, 203.0).map(f64::to_bits),
                direct
                    .map(|channel| channel.clamp(0.0, 203.0))
                    .map(f64::to_bits),
            );
        }

        // Beyond the tolerance, the ICtCp search still preserves hue.
        let outside = multiply(BT709_TO_BT2020, [-0.1, 0.0, 35.0]);
        let mapped = gamut_map_bt2020_to_bt709_ictcp_hard(outside, 203.0);
        let before = bt2020_nits_to_ictcp(outside);
        let after = bt2020_nits_to_ictcp(multiply(BT709_TO_BT2020, mapped));
        assert_near(after[2] / after[1], before[2] / before[1], 1.0e-6);
    }

    #[test]
    fn hard_ictcp_gamut_mapping_reaches_bt709_surface_at_fixed_intensity_and_hue() {
        for source in [
            [203.0, 0.0, 0.0],
            [0.0, 203.0, 0.0],
            [0.0, 0.0, 203.0],
            [203.0, 203.0, 0.0],
            [203.0, 0.0, 203.0],
            [0.0, 203.0, 203.0],
        ] {
            let direct = multiply(BT2020_TO_BT709, source);
            assert!(!rgb_is_within_cube(direct, 203.0));

            let mapped = gamut_map_bt2020_to_bt709_ictcp_hard(source, 203.0);
            assert!(rgb_is_within_cube(mapped, 203.0));
            let boundary_distance = mapped
                .into_iter()
                .map(|channel| channel.min(203.0 - channel))
                .fold(f64::INFINITY, f64::min);
            assert!(boundary_distance < 0.1);

            let before = bt2020_nits_to_ictcp(source);
            let after = bt2020_nits_to_ictcp(multiply(BT709_TO_BT2020, mapped));
            assert_near(after[0], before[0], 2.0e-8);
            let before_chroma = before[1].hypot(before[2]);
            let after_chroma = after[1].hypot(after[2]);
            assert!(after_chroma < before_chroma);
            assert_near(after[1] / after_chroma, before[1] / before_chroma, 2.0e-8);
            assert_near(after[2] / after_chroma, before[2] / before_chroma, 2.0e-8);
        }
    }

    #[test]
    fn invalid_negative_and_out_of_gamut_inputs_are_bounded() {
        let transform = transform(OutputEncoding::SdrSrgbHardware, 1_000.0, 203.0);
        for input in [
            [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
            [-1.0, 0.5, 20.0, 2.0],
        ] {
            let output = transform.transform(input);
            assert!(output.iter().all(|value| value.is_finite()));
            assert!(output.iter().all(|value| (0.0..=1.0).contains(value)));
        }
    }

    #[test]
    fn alpha_compositing_occurs_in_linear_light() {
        let composite = composite_linear([1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0]);
        for (actual, expected) in composite.into_iter().zip([0.5, 0.0, 0.5]) {
            assert_near(actual, expected, EPSILON);
        }
        assert_eq!(
            composite_linear([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).map(f64::to_bits),
            [0.0, 0.0, 1.0].map(f64::to_bits),
        );
        assert_eq!(
            composite_linear([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0]).map(f64::to_bits),
            [1.0, 0.0, 0.0].map(f64::to_bits),
        );
    }

    #[test]
    fn negative_image_color_preserves_composites_and_luminance() {
        let composite = composite_linear([-0.25, 0.5, 0.5, 0.5], [1.0; 3]);
        assert_eq!(
            composite.map(f64::to_bits),
            [0.375_f64, 0.75, 0.75].map(f64::to_bits)
        );
        let rgb = [-0.25, 0.5, 0.5];
        // Independent line-to-neutral projection into BT.2020. Both gamut
        // paths must retain the chroma direction represented by negative red.
        let y = -0.25 * 0.2627 + 0.5 * 0.6780 + 0.5 * 0.0593;
        let scale = y / (y + 0.25);
        let projected = [0.0, y + (0.5 - y) * scale, y + (0.5 - y) * scale];
        for encoding in [OutputEncoding::Pq, OutputEncoding::SdrSrgbHardware] {
            let transform = transform(encoding, 203.0, 203.0);
            let actual = transform.transform([rgb[0], rgb[1], rgb[2], 1.0]);
            let expected = transform.transform([projected[0], projected[1], projected[2], 1.0]);
            for channel in 0..3 {
                assert_near(actual[channel], expected[channel], 1e-10);
            }
            let clipped = transform.transform([0.0, 0.5, 0.5, 1.0]);
            assert!(actual[1] < clipped[1]);
            assert!(actual[2] < clipped[2]);
        }
    }

    #[test]
    fn default_target_linear_highlights_tone_map_white_from_the_observed_peak() {
        let transform = transform(OutputEncoding::SdrSrgbHardware, 812.0, 203.0);
        let white = transform.transform([1.0; 4]);
        assert_near(white[0], 0.81, 0.01);
        assert_near(transform.transform([4.0, 4.0, 4.0, 1.0])[0], 1.0, 1e-10);
    }

    #[test]
    fn deterministic_dither_is_bounded_and_zero_mean() {
        assert_eq!(
            dither_unorm(0.5, 10, 20, 1, 8).to_bits(),
            dither_unorm(0.5, 10, 20, 1, 8).to_bits()
        );
        let mean = (0..256)
            .map(|x| dither_unorm(0.5, x, 7, 0, 8) - 0.5)
            .sum::<f64>()
            / 256.0;
        assert!(mean.abs() < 0.000_5);
        for value in [-1.0, 0.0, 1.0, 2.0] {
            assert!((0.0..=1.0).contains(&dither_unorm(value, 0, 0, 0, 10)));
        }
    }
}
