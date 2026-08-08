//! CPU reference implementation of xl-view's display color pipeline.
//!
//! Decoded RGB is linear BT.2020/D65. A working value of `1.0` is the
//! fixed [`HDR_REFERENCE_WHITE_NITS`], while [`SourceIntensityTarget`] and
//! [`OutputPeak`] are always absolute cd/m².

use std::f64::consts::LN_2;

/// Conventional scRGB reference white used by wgpu's extended-linear sRGB
/// surface convention.
pub const SCRGB_REFERENCE_WHITE_NITS: f64 = 80.0;

/// BT.2100/BT.2408 HDR reference white used to normalize the working space.
pub const HDR_REFERENCE_WHITE_NITS: f32 = 203.0;

const BT2020_LUMA: [f64; 3] = [0.2627, 0.6780, 0.0593];
const BT709_LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

// PQ-based BT.2100-3 ICtCp matrices. PQ gives the absolute-nits working space
// a fixed perceptual intensity axis for both PQ and HLG sources; an HLG-based
// intermediate would require choosing a display peak and system gamma.
// LMS_P denotes PQ-encoded LMS.
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
const LMS_P_TO_ICTCP: [[f64; 3]; 3] = [
    [2_048.0 / 4_096.0, 2_048.0 / 4_096.0, 0.0],
    [6_610.0 / 4_096.0, -13_613.0 / 4_096.0, 7_003.0 / 4_096.0],
    [17_933.0 / 4_096.0, -17_390.0 / 4_096.0, -543.0 / 4_096.0],
];
const ICTCP_TO_LMS_P: [[f64; 3]; 3] = [
    [1.0, 0.008_609_037_037_932_756, 0.111_029_625_003_025_96],
    [1.0, -0.008_609_037_037_932_756, -0.111_029_625_003_025_96],
    [1.0, 0.560_031_335_710_679_1, -0.320_627_174_987_318_85],
];

/// Linear BT.709/sRGB to CIE XYZ (D65), with matrix rows stored in order.
pub const BT709_TO_XYZ_D65: [[f64; 3]; 3] = [
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
pub const XYZ_D65_TO_BT709: [[f64; 3]; 3] = [
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
pub const BT2020_TO_XYZ_D65: [[f64; 3]; 3] = [
    [
        0.636_958_048_301_291_4,
        0.144_616_903_586_208_32,
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
pub const XYZ_D65_TO_BT2020: [[f64; 3]; 3] = [
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

/// Linear BT.2020 to linear BT.709/sRGB, both D65.
pub const BT2020_TO_BT709: [[f64; 3]; 3] = [
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

/// Linear BT.709/sRGB to linear BT.2020, both D65.
pub const BT709_TO_BT2020: [[f64; 3]; 3] = [
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
positive_nits_type!(OutputPeak, "Output encoding or SDR mapping peak in cd/m².");

/// Transfer-function class of the decoded source image.
///
/// Decoded pixels are always linear BT.2020, but SDR remains relative while
/// PQ and HLG pixels carry absolute luminance normalized by HDR reference white.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceDynamicRange {
    Sdr,
    Pq,
    Hlg,
}

impl SourceDynamicRange {
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        matches!(self, Self::Pq | Self::Hlg)
    }

    #[must_use]
    pub const fn shader_code(self) -> u32 {
        match self {
            Self::Sdr => 0,
            Self::Pq => 1,
            Self::Hlg => 2,
        }
    }
}

impl SourceIntensityTarget {
    /// Conservative fallback when source metadata is unavailable or invalid.
    pub const FALLBACK: Self = Self(1_000.0);

    /// Uses valid JPEG XL metadata or the documented 1,000-nit fallback.
    #[must_use]
    pub fn from_jxl_metadata(nits: f32) -> Self {
        Self::new(f64::from(nits)).unwrap_or(Self::FALLBACK)
    }
}

/// Destination output encoding. Values returned for `SdrSrgbHardware` are
/// linear; an sRGB texture view must encode them exactly once.
///
/// `ExtendedLinear` deliberately does not tone-map or clamp positive values to
/// the display peak. It converts the canonical BT.2020 signal to linear sRGB
/// primaries using the scRGB 80-nit convention and leaves final display mapping
/// to the compositor. Consequently, HDR reference white is
/// `203 / 80 = 2.5375`, and HDR highlights may be much greater than `1.0`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputEncoding {
    Pq,
    Hlg,
    ExtendedLinear,
    SdrSrgbHardware,
    SdrSrgbExplicit,
}

/// Parameters which do not belong to source image metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputTransform {
    pub encoding: OutputEncoding,
    pub source_dynamic_range: SourceDynamicRange,
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
            .map(sanitize)
            .map(|channel| channel * exposure);
        let alpha = sanitize(rgba[3]).clamp(0.0, 1.0);
        let output = match self.encoding {
            OutputEncoding::Pq => self.to_pq(working),
            OutputEncoding::Hlg => self.to_hlg(working),
            OutputEncoding::ExtendedLinear => Self::to_extended_linear(working),
            OutputEncoding::SdrSrgbHardware => self.to_sdr(working),
            OutputEncoding::SdrSrgbExplicit => self.to_sdr(working).map(srgb_oetf),
        };
        [output[0], output[1], output[2], alpha]
    }

    fn bounded_bt2020_nits(self, working: [f64; 3]) -> [f64; 3] {
        let nits = working.map(|channel| channel * f64::from(HDR_REFERENCE_WHITE_NITS));
        gamut_map_rgb(nits, BT2020_LUMA, Some(self.output_peak.nits()))
    }

    fn mapped_bt2020_nits(self, working: [f64; 3]) -> [f64; 3] {
        if self.source_dynamic_range == SourceDynamicRange::Sdr {
            return self.bounded_bt2020_nits(working);
        }
        let reference_white = f64::from(HDR_REFERENCE_WHITE_NITS);
        let nits = working.map(|channel| channel * reference_white);
        // Keep metadata fixed so exposure moves pixels through the curve
        // instead of renormalizing its input range.
        let source_peak = self.source_intensity_target.nits();
        let output_peak = self.output_peak.nits();
        tone_map_bt2020_ictcp(nits, reference_white, source_peak, output_peak)
    }

    #[allow(clippy::cast_possible_truncation)] // The presentation shader and surface use f32.
    fn to_pq(self, working: [f64; 3]) -> [f64; 3] {
        self.bounded_bt2020_nits(working)
            .map(|nits| f64::from(pq_oetf((nits / 10_000.0) as f32)))
    }

    #[allow(clippy::cast_possible_truncation)] // The presentation shader and surface use f32.
    fn to_hlg(self, working: [f64; 3]) -> [f64; 3] {
        let peak = self.output_peak.nits();
        let display = self
            .bounded_bt2020_nits(working)
            .map(|channel| channel / peak);
        let display_luminance = dot(display, BT2020_LUMA).clamp(0.0, 1.0);
        let gamma = f64::from(hlg_system_gamma(peak as f32));
        let scene = if display_luminance > 0.0 {
            let ootf = display_luminance.powf((gamma - 1.0) / gamma);
            display.map(|channel| channel / ootf)
        } else {
            [0.0; 3]
        };
        scene.map(|channel| f64::from(hlg_oetf(channel as f32)))
    }

    fn to_extended_linear(working: [f64; 3]) -> [f64; 3] {
        // scRGB is the intentional exception to destination tone mapping. Its
        // floating-point surface carries unclamped luminance to the compositor.
        let bt709 = multiply(BT2020_TO_BT709, working)
            .map(|channel| channel * f64::from(HDR_REFERENCE_WHITE_NITS));
        gamut_map_rgb(bt709, BT709_LUMA, None).map(|nits| nits / SCRGB_REFERENCE_WHITE_NITS)
    }

    fn to_sdr(self, working: [f64; 3]) -> [f64; 3] {
        if self.source_dynamic_range == SourceDynamicRange::Sdr {
            let bt709 = multiply(BT2020_TO_BT709, working);
            return gamut_map_rgb(bt709, BT709_LUMA, Some(1.0));
        }
        let bt2020 = self.mapped_bt2020_nits(working);
        let bt709 = multiply(BT2020_TO_BT709, bt2020);
        // Resolve gamut once in destination primaries; channel clipping avoids
        // adding neutral components to saturated highlights.
        bt709.map(|nits| (nits / self.output_peak.nits()).clamp(0.0, 1.0))
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

/// BT.2100 HLG OETF from scene-linear light.
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
    let encoded = encoded.max(0.0);
    if encoded <= 0.5 {
        encoded * encoded / 3.0
    } else {
        (((encoded - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

/// BT.2100 HLG system gamma for a display peak in cd/m².
#[must_use]
#[inline]
pub fn hlg_system_gamma(display_peak_nits: f32) -> f32 {
    1.2 + 0.42 * (display_peak_nits.max(1.0) / 1_000.0).log10()
}

/// Applies a 3x3 row-major matrix to an RGB/XYZ vector.
#[must_use]
pub fn multiply(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| dot(row, vector))
}

/// Premultiplies and composites straight-alpha foreground RGB in linear light.
#[must_use]
pub fn composite_linear(foreground: [f64; 4], background: [f64; 3]) -> [f64; 3] {
    let alpha = sanitize(foreground[3]).clamp(0.0, 1.0);
    std::array::from_fn(|index| {
        sanitize(foreground[index]).mul_add(alpha, sanitize(background[index]) * (1.0 - alpha))
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

// The caller bypasses tone mapping when the source already fits the output.
fn tone_map_luminance(
    luminance: f64,
    reference_white: f64,
    source_peak: f64,
    output_peak: f64,
) -> f64 {
    let luminance = sanitize(luminance);
    let reference_white = reference_white.min(output_peak);
    let input_range = source_peak / reference_white;
    let output_range = output_peak / reference_white;
    let coefficient = (output_range * (1.0 + input_range) - input_range) / input_range.powi(2);
    let relative = luminance / reference_white;
    let mapped = relative * (1.0 + relative * coefficient) / (1.0 + relative);
    (mapped * reference_white).clamp(0.0, output_peak)
}

fn bt2020_nits_to_ictcp(rgb: [f64; 3]) -> [f64; 3] {
    let lms_p =
        multiply(BT2020_TO_LMS, rgb).map(|channel| pq_oetf_f64(channel.max(0.0) / 10_000.0));
    multiply(LMS_P_TO_ICTCP, lms_p)
}

fn ictcp_to_bt2020_nits(ictcp: [f64; 3]) -> [f64; 3] {
    let lms =
        multiply(ICTCP_TO_LMS_P, ictcp).map(|channel| pq_eotf_f64(channel.max(0.0)) * 10_000.0);
    multiply(LMS_TO_BT2020, lms)
}

fn tone_map_bt2020_ictcp(
    rgb_nits: [f64; 3],
    reference_white: f64,
    source_peak: f64,
    output_peak: f64,
) -> [f64; 3] {
    if source_peak <= output_peak {
        return rgb_nits;
    }
    let mut ictcp = bt2020_nits_to_ictcp(rgb_nits);
    let intensity_nits = pq_eotf_f64(ictcp[0].max(0.0)) * 10_000.0;
    let mapped_intensity =
        tone_map_luminance(intensity_nits, reference_white, source_peak, output_peak);
    ictcp[0] = pq_oetf_f64(mapped_intensity / 10_000.0);
    ictcp_to_bt2020_nits(ictcp)
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

fn pq_oetf_f64(normalized_luminance: f64) -> f64 {
    let m1 = f64::from(PQ_M1);
    let m2 = f64::from(PQ_M2);
    let c1 = f64::from(PQ_C1);
    let c2 = f64::from(PQ_C2);
    let c3 = f64::from(PQ_C3);
    let power = normalized_luminance.max(0.0).powf(m1);
    ((c1 + c2 * power) / (1.0 + c3 * power)).powf(m2)
}

// Hue-preserving destination-gamut mapping: move along the line from the
// target-space neutral of equal luminance until every channel is representable.
fn gamut_map_rgb(rgb: [f64; 3], luma: [f64; 3], upper: Option<f64>) -> [f64; 3] {
    // Preserve finite negative matrix results here: they carry the chroma
    // direction needed for line-to-neutral gamut compression.
    let rgb = rgb.map(finite_or_zero);
    let maximum = upper.unwrap_or(f64::INFINITY);
    let neutral = dot(rgb, luma).clamp(0.0, maximum);
    let mut scale = 1.0_f64;
    for channel in rgb {
        let chroma = channel - neutral;
        if chroma < 0.0 {
            scale = scale.min(neutral / -chroma);
        } else if chroma > 0.0 && maximum.is_finite() {
            scale = scale.min((maximum - neutral) / chroma);
        }
    }
    rgb.map(|channel| (neutral + (channel - neutral) * scale.clamp(0.0, 1.0)).clamp(0.0, maximum))
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
    if stops.is_finite() {
        (stops * LN_2).exp()
    } else {
        1.0
    }
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
            source_dynamic_range: SourceDynamicRange::Hlg,
            source_intensity_target: SourceIntensityTarget::new(source).unwrap(),
            output_peak: OutputPeak::new(output).unwrap(),
            exposure_stops: 0.0,
        }
    }

    fn sdr_transform(encoding: OutputEncoding, output: f64) -> OutputTransform {
        OutputTransform {
            source_dynamic_range: SourceDynamicRange::Sdr,
            source_intensity_target: SourceIntensityTarget::new(255.0).unwrap(),
            ..transform(encoding, 255.0, output)
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

        for value in [0.0_f32, 1.0 / 12.0, 0.18, 1.0] {
            assert_near(hlg_inverse_oetf(hlg_oetf(value)), value, 1.0e-6);
        }
        assert_near(hlg_oetf(1.0 / 12.0), 0.5, f64::from(f32::EPSILON));
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
    fn ictcp_matrices_round_trip_bt2020_and_preserve_neutral_intensity() {
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
    fn modified_reinhard_curve_is_smooth_and_hits_declared_endpoints() {
        let reference = 203.0;
        let source_peak = 1_000.0;
        let output_peak = 203.0;
        assert_near(
            tone_map_luminance(0.0, reference, source_peak, output_peak),
            0.0,
            EPSILON,
        );
        assert_near(
            tone_map_luminance(source_peak, reference, source_peak, output_peak),
            output_peak,
            EPSILON,
        );

        let mut previous = -1.0;
        for input in 0..=1_000 {
            let mapped = tone_map_luminance(f64::from(input), reference, source_peak, output_peak);
            assert!(mapped >= previous);
            assert!(mapped <= output_peak);
            previous = mapped;
        }
        assert_near(
            tone_map_luminance(reference, reference, source_peak, output_peak),
            105.682_713_5,
            1.0e-10,
        );
    }

    #[test]
    fn ictcp_tone_mapping_retains_perceptual_chroma_components() {
        let source = [1_000.0, 250.0, 40.0];
        let before = bt2020_nits_to_ictcp(source);
        let mapped = tone_map_bt2020_ictcp(source, 203.0, 1_000.0, 203.0);
        let after = bt2020_nits_to_ictcp(mapped);
        assert_near(after[1], before[1], 2.0e-6);
        assert_near(after[2], before[2], 2.0e-6);
        assert!(after[0] < before[0]);
    }

    #[test]
    fn hdr_to_sdr_clips_final_channels_without_adding_neutral() {
        let transform = transform(OutputEncoding::SdrSrgbHardware, 1_000.0, 203.0);
        let peak = 1_000.0 / 203.0;
        for (input, expected) in [
            ([peak, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
            ([0.0, peak, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]),
            ([0.0, 0.0, peak, 1.0], [0.0, 0.0, 1.0, 1.0]),
        ] {
            let output = transform.transform(input);
            for index in 0..4 {
                assert_near(output[index], expected[index], EPSILON);
            }
        }
    }

    #[test]
    fn source_peak_only_drives_sdr_mapping_and_hlg_uses_its_container_peak() {
        assert_near(hlg_system_gamma(1_000.0), 1.2, f64::from(f32::EPSILON));
        assert!(hlg_system_gamma(2_000.0) > hlg_system_gamma(1_000.0));

        let pixel = [2.0, 2.0, 2.0, 1.0];
        let sdr_4k = transform(OutputEncoding::SdrSrgbHardware, 4_000.0, 203.0).transform(pixel);
        let sdr_100 = transform(OutputEncoding::SdrSrgbHardware, 100.0, 203.0).transform(pixel);
        assert!((sdr_4k[0] - sdr_100[0]).abs() > EPSILON);

        let hlg_4k = transform(OutputEncoding::Hlg, 4_000.0, 1_000.0).transform(pixel);
        let hlg_1k = transform(OutputEncoding::Hlg, 1_000.0, 1_000.0).transform(pixel);
        assert_near(hlg_4k[0], hlg_1k[0], EPSILON);

        let hlg_2k_container = transform(OutputEncoding::Hlg, 4_000.0, 2_000.0).transform(pixel);
        assert!((hlg_4k[0] - hlg_2k_container[0]).abs() > EPSILON);
    }

    #[test]
    fn every_output_transform_is_finite_and_neutral() {
        for encoding in [
            OutputEncoding::Pq,
            OutputEncoding::Hlg,
            OutputEncoding::ExtendedLinear,
            OutputEncoding::SdrSrgbHardware,
            OutputEncoding::SdrSrgbExplicit,
        ] {
            let output = transform(encoding, 4_000.0, 1_000.0).transform([1.0, 1.0, 1.0, 0.5]);
            assert!(output.iter().all(|value| value.is_finite()));
            assert_near(output[0], output[1], 1.0e-12);
            assert_near(output[1], output[2], 1.0e-12);
            assert_near(output[3], 0.5, EPSILON);
        }
    }

    #[test]
    fn sdr_white_has_output_specific_reference_levels() {
        let sdr =
            sdr_transform(OutputEncoding::SdrSrgbHardware, 203.0).transform([1.0, 1.0, 1.0, 1.0]);
        assert_near(sdr[0], 1.0, EPSILON);

        let pq = sdr_transform(OutputEncoding::Pq, 10_000.0).transform([1.0, 1.0, 1.0, 1.0]);
        assert_near(pq[0], pq_oetf(203.0 / 10_000.0), 1.0e-7);

        let hlg = sdr_transform(OutputEncoding::Hlg, 1_000.0).transform([1.0, 1.0, 1.0, 1.0]);
        assert_near(hlg[0], 0.75, 2.0e-4);

        let scrgb =
            sdr_transform(OutputEncoding::ExtendedLinear, 1_000.0).transform([1.0, 1.0, 1.0, 1.0]);
        assert_near(scrgb[0], 203.0 / SCRGB_REFERENCE_WHITE_NITS, 2.0e-14);
    }

    #[test]
    fn compositor_managed_pq_preserves_absolute_hdr_luminance() {
        let transform = transform(OutputEncoding::Pq, 4_000.0, 10_000.0);
        let output = transform.transform([1_000.0 / 203.0; 4]);
        assert_near(output[0], pq_oetf(1_000.0 / 10_000.0), 1.0e-7);
    }

    #[test]
    fn tone_mapping_is_monotonic_rolls_highlights_and_reaches_sdr_peak() {
        let transform = transform(OutputEncoding::SdrSrgbHardware, 4_000.0, 203.0);
        let mut previous = -1.0;
        for step in 0..=4_000 {
            let value = f64::from(step) / 203.0;
            let output = transform.transform([value; 4])[0];
            assert!(output >= previous);
            assert!((0.0..=1.0).contains(&output));
            previous = output;
        }
        assert!(transform.transform([1_000.0 / 203.0; 4])[0] < 1.0);
        assert_near(transform.transform([4_000.0 / 203.0; 4])[0], 1.0, 2.0e-6);
    }

    #[test]
    fn exposure_moves_pixels_through_a_fixed_tone_curve() {
        let base = transform(OutputEncoding::SdrSrgbHardware, 4_000.0, 203.0);
        let exposed = OutputTransform {
            exposure_stops: 1.0,
            ..base
        };
        assert_near(
            exposed.transform([0.25, 0.25, 0.25, 1.0])[0],
            base.transform([0.5, 0.5, 0.5, 1.0])[0],
            EPSILON,
        );

        let peak = transform(OutputEncoding::SdrSrgbHardware, 1_000.0, 203.0);
        let lowered = OutputTransform {
            exposure_stops: -1.0,
            ..peak
        };
        let source_white = [1_000.0 / 203.0; 4];
        assert_near(peak.transform(source_white)[0], 1.0, 2.0e-6);
        assert!(lowered.transform(source_white)[0] < 1.0);
    }

    #[test]
    fn extended_linear_is_intentionally_unclamped_and_not_tone_mapped() {
        let transform = transform(OutputEncoding::ExtendedLinear, 4_000.0, 500.0);
        let reference_white = transform.transform([1.0, 1.0, 1.0, 1.0]);
        for channel in &reference_white[..3] {
            assert_near(*channel, 203.0 / SCRGB_REFERENCE_WHITE_NITS, 2.0e-14);
        }

        let highlight = transform.transform([20.0, 20.0, 20.0, 1.0]);
        for channel in &highlight[..3] {
            assert_near(*channel, 20.0 * 203.0 / SCRGB_REFERENCE_WHITE_NITS, 3.0e-13);
            assert!(*channel > transform.output_peak.nits() / SCRGB_REFERENCE_WHITE_NITS);
        }
    }

    #[test]
    fn invalid_negative_and_out_of_gamut_inputs_are_deliberate() {
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
