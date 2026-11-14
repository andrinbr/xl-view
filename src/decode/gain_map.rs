//! ISO 21496-1 gain maps stored in JPEG XL `jhgm` boxes.
//!
//! [`CaptureInput`] copies the box out of the bytes the main decoder reads,
//! within a byte budget, because `jxl` buffers the boxes it extracts itself
//! without a limit. [`prepare`] validates the capture and decodes the gain map
//! image. [`GainMap::apply`] then turns the SDR base image's linear pixels into
//! the HDR alternate image.
//!
//! Only the forward direction is supported: an SDR base image with a
//! higher-headroom alternate image. An unsupported or broken gain map leaves
//! the base image untouched and reports a [`GainMapSkipReason`].

use std::cmp::Ordering;
use std::io::{self, IoSliceMut, Read};

use jxl::api::{
    Endianness, Event, JxlBitstreamInput, JxlColorEncoding, JxlColorProfile, JxlColorType,
    JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, JxlPrimaries,
    JxlWhitePoint, Orientation,
};
use moxcms::{ColorProfile, Layout, TransformOptions};

use super::jxl::{RayonRunner, profile_intent, working_profile};
use super::{DecodeLimits, GainMapSkipReason, GainMapStatus, Rendition, SourceDynamicRange};
use crate::units::usize_from_u32;

const SIGNATURE_BOX_HEADER: [u8; 8] = *b"\0\0\0\x0cJXL ";
const GAIN_MAP_BOX: [u8; 4] = *b"jhgm";
const BROTLI_BOX: [u8; 4] = *b"brob";

const FLAG_MULTICHANNEL: u8 = 0x80;
const FLAG_BASE_COLOR_SPACE: u8 = 0x40;

/// Gain map samples are decoded as 16-bit codes.
const MAX_CODE: f32 = 65_535.0;
/// Largest finite f16, the range of the canonical pixel format.
const MAX_OUTPUT: f32 = 65_504.0;
/// 2^64 keeps both matrix products finite in f32 without clipping any value
/// that fits the canonical format.
const INTERMEDIATE_LIMIT: f32 = 18_446_744_073_709_551_616.0;
/// 2^32 bounds each matrix row so a product with `INTERMEDIATE_LIMIT` stays finite.
const MAX_ROW_SUM: f64 = 4_294_967_296.0;
/// Bound on ‖A‖∞ × ‖B‖∞. Real color spaces stay below 3 relative to BT.2020.
const MAX_CONDITION: f64 = 64.0;
const MAX_INVERSE_RESIDUAL: f64 = 1.0 / 65_536.0;

type Reason = GainMapSkipReason;
type Matrix = [[f32; 3]; 3];

/// The bytes of one `jhgm` box, or of the `brob` box that wraps it.
pub(super) struct Payload {
    data: Vec<u8>,
    compressed: bool,
}

/// What the container scan found.
pub(super) enum Capture {
    /// No complete gain map box was seen.
    None,
    Payload(Payload),
    Skipped(Reason),
}

enum Scan {
    /// Collecting a box header.
    Header,
    /// Inside a box payload. `None` extends to the end of the file.
    Payload(Option<u64>),
    /// A bare codestream, or framing the decoder rejects itself.
    Stopped,
}

enum Found {
    Nothing,
    Capturing(Payload),
    Complete(Payload),
    Skipped(Reason),
}

/// Forwards the main decoder's input unchanged and copies the gain map box
/// out of it.
///
/// `jxl` is not asked for the box, so it skips the payload. Skipping uses the
/// provided [`JxlBitstreamInput::skip`], which reads, so every byte still
/// passes through [`Self::read`].
pub(super) struct CaptureInput<'a, I> {
    inner: &'a mut I,
    scan: Scan,
    /// 8 or 16 header bytes, plus the wrapped type of a `brob` box.
    header: [u8; 20],
    header_len: usize,
    first_box: bool,
    end_of_input: bool,
    limit: usize,
    found: Found,
}

impl<'a, I: JxlBitstreamInput> CaptureInput<'a, I> {
    /// `limit` bounds the capture until [`Self::set_budget`] narrows it.
    pub(super) fn new(inner: &'a mut I, limit: usize) -> Self {
        Self {
            inner,
            scan: Scan::Header,
            header: [0; 20],
            header_len: 0,
            first_box: true,
            end_of_input: false,
            limit,
            found: Found::Nothing,
        }
    }

    /// Applies the exact budget once the base image's size is known. A box
    /// that arrived earlier and exceeds it is discarded.
    pub(super) fn set_budget(&mut self, budget: usize) {
        self.limit = self.limit.min(budget);
        if let Found::Capturing(payload) | Found::Complete(payload) = &self.found
            && payload.data.len() > self.limit
        {
            self.found = Found::Skipped(Reason::TooLarge);
        }
    }

    /// Ends the scan. With `read_to_end`, a gain map box that extends to the
    /// end of the file is read completely first: `jxl` stops reading at such
    /// a box because it did not request it.
    pub(super) fn finish(&mut self, read_to_end: bool) -> io::Result<Capture> {
        let mut buffer = [0_u8; 8192];
        while read_to_end
            && !self.end_of_input
            && matches!(self.scan, Scan::Payload(None))
            && matches!(self.found, Found::Capturing(_))
        {
            match self.read(&mut [IoSliceMut::new(&mut buffer)]) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                result => {
                    result?;
                }
            }
        }
        Ok(match std::mem::replace(&mut self.found, Found::Nothing) {
            Found::Complete(payload) => Capture::Payload(payload),
            Found::Capturing(payload)
                if self.end_of_input && matches!(self.scan, Scan::Payload(None)) =>
            {
                Capture::Payload(payload)
            }
            Found::Skipped(reason) => Capture::Skipped(reason),
            Found::Nothing | Found::Capturing(_) => Capture::None,
        })
    }

    fn header_needed(&self) -> usize {
        if self.header_len < 8 {
            return 8;
        }
        let large_size = if self.header[..4] == [0, 0, 0, 1] {
            8
        } else {
            0
        };
        let wrapped_type = if self.header[4..8] == BROTLI_BOX {
            4
        } else {
            0
        };
        8 + large_size + wrapped_type
    }

    fn scan_bytes(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            match self.scan {
                Scan::Stopped => return,
                Scan::Header => {
                    let take = (self.header_needed() - self.header_len).min(bytes.len());
                    self.header[self.header_len..self.header_len + take]
                        .copy_from_slice(&bytes[..take]);
                    self.header_len += take;
                    bytes = &bytes[take..];
                    if self.header_len == self.header_needed() {
                        self.begin_box();
                    }
                }
                Scan::Payload(remaining) => {
                    let take = remaining
                        .and_then(|remaining| usize::try_from(remaining).ok())
                        .map_or(bytes.len(), |remaining| remaining.min(bytes.len()));
                    self.capture_bytes(&bytes[..take]);
                    bytes = &bytes[take..];
                    if let Some(remaining) = remaining {
                        self.scan = Scan::Payload(Some(remaining - take as u64));
                        self.end_box_if_complete();
                    }
                }
            }
        }
    }

    fn begin_box(&mut self) {
        let large_size = self.header[..4] == [0, 0, 0, 1];
        let header_size = if large_size { 16 } else { 8 };
        let wrapped = self.header[4..8] == BROTLI_BOX;
        let consumed = header_size + if wrapped { 4 } else { 0 };
        let declared = if large_size {
            u64::from_be_bytes(self.header[8..16].try_into().expect("eight size bytes"))
        } else {
            u64::from(u32::from_be_bytes(
                self.header[..4].try_into().expect("four size bytes"),
            ))
        };
        let type_offset = if wrapped { header_size } else { 4 };
        let box_type: [u8; 4] = self.header[type_offset..type_offset + 4]
            .try_into()
            .expect("four type bytes");
        self.header_len = 0;
        if std::mem::take(&mut self.first_box) && self.header[..8] != SIGNATURE_BOX_HEADER {
            self.scan = Scan::Stopped;
            return;
        }
        let remaining = if declared == 0 {
            None
        } else if let Some(remaining) = declared.checked_sub(consumed as u64) {
            Some(remaining)
        } else {
            self.scan = Scan::Stopped;
            return;
        };
        if box_type == GAIN_MAP_BOX {
            self.begin_capture(remaining, wrapped);
        }
        self.scan = Scan::Payload(remaining);
        self.end_box_if_complete();
    }

    fn begin_capture(&mut self, remaining: Option<u64>, compressed: bool) {
        self.found = if !matches!(self.found, Found::Nothing) {
            // Nothing says which of several gain maps applies.
            Found::Skipped(Reason::Unsupported)
        } else if remaining.is_some_and(|remaining| remaining > self.limit as u64) {
            Found::Skipped(Reason::TooLarge)
        } else {
            Found::Capturing(Payload {
                data: Vec::new(),
                compressed,
            })
        };
    }

    fn capture_bytes(&mut self, bytes: &[u8]) {
        let Found::Capturing(payload) = &mut self.found else {
            return;
        };
        if payload.data.len() + bytes.len() > self.limit {
            self.found = Found::Skipped(Reason::TooLarge);
        } else if payload.data.try_reserve(bytes.len()).is_err() {
            self.found = Found::Skipped(Reason::OutOfMemory);
        } else {
            payload.data.extend_from_slice(bytes);
        }
    }

    fn end_box_if_complete(&mut self) {
        if !matches!(self.scan, Scan::Payload(Some(0))) {
            return;
        }
        self.scan = Scan::Header;
        if matches!(self.found, Found::Capturing(_))
            && let Found::Capturing(payload) = std::mem::replace(&mut self.found, Found::Nothing)
        {
            self.found = Found::Complete(payload);
        }
    }
}

impl<I: JxlBitstreamInput> JxlBitstreamInput for CaptureInput<'_, I> {
    fn available_bytes(&mut self) -> io::Result<usize> {
        self.inner.available_bytes()
    }

    fn read(&mut self, buffers: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let count = self.inner.read(buffers)?;
        if count == 0 {
            self.end_of_input |= buffers.iter().any(|buffer| !buffer.is_empty());
            return Ok(0);
        }
        let mut unscanned = count;
        for buffer in buffers.iter() {
            let filled = unscanned.min(buffer.len());
            self.scan_bytes(&buffer[..filled]);
            unscanned -= filled;
            if unscanned == 0 {
                break;
            }
        }
        Ok(count)
    }
}

/// Facts about the base image that decide whether its gain map applies.
pub(super) struct BaseImage {
    pub(super) width: u32,
    pub(super) height: u32,
    /// The enumerated RGB encoding the decoder output, when it equals the
    /// embedded one. `None` for ICC, XYB fallback and grayscale output.
    pub(super) encoding: Option<JxlColorEncoding>,
    pub(super) dynamic_range: SourceDynamicRange,
    pub(super) identity_orientation: bool,
    pub(super) animated: bool,
}

/// Gains need known primaries: an enumerated RGB encoding that the decoder
/// output exactly as it is embedded.
pub(super) fn base_encoding(
    output: &JxlColorProfile,
    embedded: Option<&JxlColorProfile>,
) -> Option<JxlColorEncoding> {
    match output {
        JxlColorProfile::Simple(encoding @ JxlColorEncoding::RgbColorSpace { .. })
            if embedded == Some(output) =>
        {
            Some(encoding.clone())
        }
        _ => None,
    }
}

/// Validates a captured gain map and, for the HDR rendition, decodes it.
///
/// `budget` bounds the box payload, its decompressed form and the decoded
/// samples together. `rendition` is read once, after every check that does
/// not depend on it, so both renditions classify a file the same way.
pub(super) fn prepare(
    capture: Capture,
    base: &BaseImage,
    budget: usize,
    limits: DecodeLimits,
    rendition: &dyn Fn() -> Rendition,
) -> (Option<GainMap>, Option<GainMapStatus>) {
    let payload = match capture {
        Capture::None => return (None, None),
        Capture::Skipped(reason) => return (None, Some(GainMapStatus::NotApplied(reason))),
        Capture::Payload(payload) => payload,
    };
    match reconstruction(payload, base, budget, limits, rendition) {
        Ok(Some(gain_map)) => {
            let status = GainMapStatus::Applied {
                headroom_stops: gain_map.headroom_stops,
            };
            (Some(gain_map), Some(status))
        }
        Ok(None) => (None, Some(GainMapStatus::Available)),
        Err(reason) => (None, Some(GainMapStatus::NotApplied(reason))),
    }
}

fn reconstruction(
    payload: Payload,
    base: &BaseImage,
    budget: usize,
    limits: DecodeLimits,
    rendition: &dyn Fn() -> Rendition,
) -> Result<Option<GainMap>, Reason> {
    if base.animated {
        return Err(Reason::Unsupported);
    }
    let mut used = payload.data.len();
    let bundle = if payload.compressed {
        let bundle = decompress(&payload.data, budget.saturating_sub(used))?;
        used += bundle.len();
        bundle
    } else {
        payload.data
    };
    let bundle = Bundle::parse(&bundle)?;
    let metadata = Metadata::parse(bundle.metadata)?;
    match metadata
        .alternate_headroom
        .partial_cmp(&metadata.base_headroom)
    {
        Some(Ordering::Greater) => {}
        Some(Ordering::Equal) => return Err(Reason::NoHeadroom),
        Some(Ordering::Less) => return Err(Reason::HdrBase),
        None => return Err(Reason::Malformed),
    }
    match base.dynamic_range {
        SourceDynamicRange::Sdr => {}
        SourceDynamicRange::Pq | SourceDynamicRange::Hlg => return Err(Reason::HdrBase),
        SourceDynamicRange::Linear => return Err(Reason::Unsupported),
    }
    // Nothing defines how a rotated base image aligns with its gain map.
    if !base.identity_orientation {
        return Err(Reason::Unsupported);
    }
    let base_encoding = base.encoding.as_ref().ok_or(Reason::UnsupportedProfile)?;
    let space = if metadata.base_color_space {
        GainSpace::for_encoding(base_encoding)?
    } else {
        let alternate = bundle.alternate_encoding()?;
        GainSpace::for_encoding(alternate.as_ref().unwrap_or(base_encoding))?
    };
    let decoder = MapDecoder::open(bundle.codestream, limits)?;
    if decoder
        .storage_bytes()
        .and_then(|storage| storage.checked_add(used))
        .is_none_or(|total| total > budget)
    {
        return Err(Reason::TooLarge);
    }
    if rendition() == Rendition::Sdr {
        return Ok(None);
    }
    let samples = decoder.decode()?;
    Ok(Some(GainMap::new(samples, &metadata, space, base)))
}

fn decompress(compressed: &[u8], limit: usize) -> Result<Vec<u8>, Reason> {
    let mut decoder = brotli_decompressor::Decompressor::new(compressed, 4096);
    let mut output = Vec::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let count = match decoder.read(&mut buffer) {
            Ok(0) => return Ok(output),
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(Reason::Malformed),
        };
        if output.len() + count > limit {
            return Err(Reason::TooLarge);
        }
        output.try_reserve(count).map_err(|_| Reason::OutOfMemory)?;
        output.extend_from_slice(&buffer[..count]);
    }
}

/// Big-endian reads that report truncation as malformed data.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], Reason> {
        let (head, tail) = self.0.split_at_checked(count).ok_or(Reason::Malformed)?;
        self.0 = tail;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], Reason> {
        Ok(self.take(N)?.try_into().expect("the length was checked"))
    }

    fn u8(&mut self) -> Result<u8, Reason> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, Reason> {
        Ok(u16::from_be_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, Reason> {
        Ok(u32::from_be_bytes(self.array()?))
    }

    /// A numerator, then a denominator that must not be zero.
    fn ratio(&mut self, signed: bool) -> Result<f64, Reason> {
        let numerator = self.u32()?;
        let numerator = if signed {
            f64::from(numerator.cast_signed())
        } else {
            f64::from(numerator)
        };
        match self.u32()? {
            0 => Err(Reason::Malformed),
            denominator => Ok(numerator / f64::from(denominator)),
        }
    }
}

/// The `jhgm` box payload: version, ISO 21496-1 metadata, the alternate
/// image's color profile, then the gain map codestream.
struct Bundle<'a> {
    metadata: &'a [u8],
    color_encoding: &'a [u8],
    has_icc: bool,
    codestream: &'a [u8],
}

impl<'a> Bundle<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, Reason> {
        let mut reader = Reader(bytes);
        if reader.u8()? != 0 {
            return Err(Reason::Unsupported);
        }
        let metadata_len = usize::from(reader.u16()?);
        let metadata = reader.take(metadata_len)?;
        let color_encoding_len = usize::from(reader.u8()?);
        let color_encoding = reader.take(color_encoding_len)?;
        let icc_len = usize::try_from(reader.u32()?).map_err(|_| Reason::Malformed)?;
        let has_icc = !reader.take(icc_len)?.is_empty();
        if reader.0.is_empty() {
            return Err(Reason::Malformed);
        }
        Ok(Self {
            metadata,
            color_encoding,
            has_icc,
            codestream: reader.0,
        })
    }

    /// The alternate image's enumerated encoding, or `None` when it inherits
    /// the base image's. An ICC profile is never decoded: `jxl` allocates the
    /// size it declares, up to 256 MiB, before reading it.
    fn alternate_encoding(&self) -> Result<Option<JxlColorEncoding>, Reason> {
        if self.has_icc {
            return Err(Reason::UnsupportedProfile);
        }
        if self.color_encoding.is_empty() {
            return Ok(None);
        }
        match JxlColorProfile::from_raw_bytes(self.color_encoding, &[]) {
            Ok(JxlColorProfile::Simple(encoding @ JxlColorEncoding::RgbColorSpace { .. })) => {
                Ok(Some(encoding))
            }
            Ok(_) => Err(Reason::UnsupportedProfile),
            Err(_) => Err(Reason::Malformed),
        }
    }
}

/// ISO 21496-1 gain map metadata, version 0.
struct Metadata {
    /// Gains apply in the base image's color space, otherwise the alternate's.
    base_color_space: bool,
    base_headroom: f64,
    alternate_headroom: f64,
    channels: [Channel; 3],
}

/// One channel's parameters, as the f32 values the pixels are computed with.
#[derive(Clone, Copy)]
struct Channel {
    gain_min: f32,
    gain_max: f32,
    /// 1 / gamma.
    exponent: f32,
    base_offset: f32,
    alternate_offset: f32,
}

impl Metadata {
    fn parse(bytes: &[u8]) -> Result<Self, Reason> {
        let mut reader = Reader(bytes);
        let minimum_version = reader.u16()?;
        let _writer_version = reader.u16()?;
        let flags = reader.u8()?;
        // Reserved bits include the legacy layouts, which are not parsed.
        if minimum_version != 0 || flags & !(FLAG_MULTICHANNEL | FLAG_BASE_COLOR_SPACE) != 0 {
            return Err(Reason::Unsupported);
        }
        let base_headroom = reader.ratio(false)?;
        let alternate_headroom = reader.ratio(false)?;
        let first = Channel::parse(&mut reader)?;
        let channels = if flags & FLAG_MULTICHANNEL == 0 {
            [first; 3]
        } else {
            [
                first,
                Channel::parse(&mut reader)?,
                Channel::parse(&mut reader)?,
            ]
        };
        Ok(Self {
            base_color_space: flags & FLAG_BASE_COLOR_SPACE != 0,
            base_headroom,
            alternate_headroom,
            channels,
        })
    }
}

impl Channel {
    #[expect(clippy::cast_possible_truncation)] // Gains apply in f32, like the pixels.
    fn parse(reader: &mut Reader<'_>) -> Result<Self, Reason> {
        let gain_min = reader.ratio(true)?;
        let gain_max = reader.ratio(true)?;
        let gamma = reader.ratio(false)?;
        let channel = Self {
            gain_min: gain_min as f32,
            gain_max: gain_max as f32,
            exponent: (1.0 / gamma) as f32,
            base_offset: reader.ratio(true)? as f32,
            alternate_offset: reader.ratio(true)? as f32,
        };
        // Both multipliers must be usable f32 values, so no gain is zero or infinite.
        let usable = |gain: f32| {
            let multiplier = gain.exp2();
            multiplier.is_finite() && multiplier > 0.0
        };
        if gamma <= 0.0
            || gain_min > gain_max
            || !usable(channel.gain_min)
            || !usable(channel.gain_max)
        {
            return Err(Reason::Malformed);
        }
        Ok(channel)
    }

    /// The linear gain for a normalized gain map sample.
    ///
    /// The clamps keep every step inside the range validated while parsing,
    /// so the result is positive and finite: filtering can round a sample
    /// past 1, and the sum can round past the largest gain.
    fn multiplier(&self, sample: f32) -> f32 {
        let shaped = sample.clamp(0.0, 1.0).powf(self.exponent);
        (self.gain_min + shaped * (self.gain_max - self.gain_min))
            .clamp(self.gain_min, self.gain_max)
            .exp2()
    }
}

/// Matrices between linear BT.2020 and the linear color space gains apply in.
struct GainSpace {
    to_gain: Matrix,
    from_gain: Matrix,
}

impl GainSpace {
    /// `None` when gains apply in linear BT.2020 itself.
    fn for_encoding(encoding: &JxlColorEncoding) -> Result<Option<Self>, Reason> {
        let JxlColorEncoding::RgbColorSpace {
            white_point,
            primaries,
            ..
        } = encoding
        else {
            return Err(Reason::UnsupportedProfile);
        };
        if *white_point == JxlWhitePoint::D65 && *primaries == JxlPrimaries::BT2100 {
            return Ok(None);
        }
        working_space_matrix(encoding)
            .and_then(Self::from_matrix)
            .map(Some)
            .ok_or(Reason::UnsupportedProfile)
    }

    /// Rejects matrices that could overflow or lose accuracy. `from_gain`
    /// maps the gain color space to linear BT.2020.
    #[expect(clippy::cast_possible_truncation)] // Checked below on the stored f32 values.
    fn from_matrix(from_gain: [[f64; 3]; 3]) -> Option<Self> {
        let to_gain = inverse(&from_gain)?;
        let narrow = |matrix: [[f64; 3]; 3]| matrix.map(|row| row.map(|value| value as f32));
        let space = Self {
            to_gain: narrow(to_gain),
            from_gain: narrow(from_gain),
        };
        let norms = [row_sum_norm(&space.to_gain), row_sum_norm(&space.from_gain)];
        (norms.iter().all(|norm| *norm <= MAX_ROW_SUM)
            && norms[0] * norms[1] <= MAX_CONDITION
            && inverse_residual(&space.to_gain, &space.from_gain) <= MAX_INVERSE_RESIDUAL
            && inverse_residual(&space.from_gain, &space.to_gain) <= MAX_INVERSE_RESIDUAL)
            .then_some(space)
    }
}

/// Linear `encoding` to linear BT.2020, probed through the CMS that converts
/// the base image so both agree on the primaries.
fn working_space_matrix(encoding: &JxlColorEncoding) -> Option<[[f64; 3]; 3]> {
    let source = JxlColorProfile::Simple(encoding.with_linear_tf());
    let target = working_profile();
    let source_icc = source.try_as_icc()?;
    let target_icc = target.try_as_icc()?;
    let source = ColorProfile::new_from_slice(&source_icc).ok()?;
    let target = ColorProfile::new_from_slice(&target_icc).ok()?;
    let transform = source
        .create_transform_f32(
            Layout::Rgb,
            &target,
            Layout::Rgb,
            TransformOptions {
                rendering_intent: profile_intent(&source_icc),
                prefer_fixed_point: false,
                allow_extended_range_rgb_xyz: true,
                ..Default::default()
            },
        )
        .ok()?;
    let mut columns = [0.0_f32; 9];
    transform
        .transform(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &mut columns)
        .ok()?;
    Some(std::array::from_fn(|row| {
        std::array::from_fn(|column| f64::from(columns[column * 3 + row]))
    }))
}

/// `None` for a singular matrix. `moxcms` would substitute the identity.
#[expect(clippy::many_single_char_names)] // The usual names of a 3x3 matrix's entries.
fn inverse(matrix: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let [[a, b, c], [d, e, f], [g, h, i]] = *matrix;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if determinant == 0.0 || !determinant.is_finite() {
        return None;
    }
    let inverse = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ]
    .map(|row| row.map(|value| value / determinant));
    inverse
        .as_flattened()
        .iter()
        .all(|value| value.is_finite())
        .then_some(inverse)
}

/// The largest absolute row sum. NaN and infinity compare above every bound.
fn row_sum_norm(matrix: &Matrix) -> f64 {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| f64::from(value.abs())).sum::<f64>())
        .fold(0.0, |norm, sum| {
            if sum > norm || sum.is_nan() {
                sum
            } else {
                norm
            }
        })
}

/// ‖left · right − I‖∞.
fn inverse_residual(left: &Matrix, right: &Matrix) -> f64 {
    (0..3)
        .map(|row| {
            (0..3)
                .map(|column| {
                    let product: f64 = (0..3)
                        .map(|k| f64::from(left[row][k]) * f64::from(right[k][column]))
                        .sum();
                    (product - f64::from(u8::from(row == column))).abs()
                })
                .sum::<f64>()
        })
        .fold(0.0, f64::max)
}

fn multiply(matrix: &Matrix, vector: [f32; 3]) -> [f32; 3] {
    matrix.map(|row| row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2])
}

fn decoder_reason(error: &jxl::error::Error) -> Reason {
    match error {
        jxl::error::Error::OutOfMemory(..) | jxl::error::Error::ImageOutOfMemory(..) => {
            Reason::OutOfMemory
        }
        jxl::error::Error::ImageSizeTooLarge(..) => Reason::TooLarge,
        _ => Reason::Malformed,
    }
}

// The codestream is complete in memory, so a request for more input means truncation.
fn advance(
    decoder: &mut JxlDecoder,
    input: &mut &[u8],
    buffers: Option<&mut [JxlOutputBuffer<'_>]>,
) -> Result<Event, Reason> {
    match decoder
        .process(input, buffers, Some(&mut RayonRunner))
        .map_err(|error| decoder_reason(&error))?
    {
        Event::NeedMoreInput { .. } => Err(Reason::Malformed),
        event => Ok(event),
    }
}

/// A gain map codestream whose header passed every check.
struct MapDecoder<'a> {
    decoder: JxlDecoder,
    input: &'a [u8],
    width: usize,
    height: usize,
    channels: usize,
}

struct Samples {
    data: Vec<u16>,
    width: usize,
    height: usize,
    channels: usize,
}

impl<'a> MapDecoder<'a> {
    fn open(codestream: &'a [u8], limits: DecodeLimits) -> Result<Self, Reason> {
        let mut options = JxlDecoderOptions::default();
        // Also bounds frames inside the codestream, which the storage check
        // on the image size does not cover.
        options.sample_limit = Some(limits.memory_ceiling_bytes / size_of::<f32>());
        let mut decoder = JxlDecoder::new(options);
        let mut input = codestream;
        if advance(&mut decoder, &mut input, None)? != Event::BasicInfo {
            return Err(Reason::Malformed);
        }
        let info = decoder.basic_info().ok_or(Reason::Malformed)?;
        let (width, height) = info.size;
        if info.orientation != Orientation::Identity
            || info.animation.is_some()
            || !info.extra_channels.is_empty()
        {
            return Err(Reason::Unsupported);
        }
        let channels = match decoder
            .embedded_color_profile()
            .map(JxlColorProfile::channels)
        {
            Some(1) => 1,
            Some(3) => 3,
            _ => return Err(Reason::UnsupportedProfile),
        };
        decoder
            .set_pixel_format(JxlPixelFormat {
                color_type: if channels == 1 {
                    JxlColorType::Grayscale
                } else {
                    JxlColorType::Rgb
                },
                color_data_format: Some(JxlDataFormat::U16 {
                    endianness: Endianness::native(),
                    bit_depth: 16,
                }),
                extra_channel_format: Vec::new(),
            })
            .map_err(|error| decoder_reason(&error))?;
        // Samples are gains, not colors, and are used exactly as encoded. A
        // conversion to another output profile would change them.
        if decoder.output_color_profile() != decoder.embedded_color_profile() {
            return Err(Reason::UnsupportedProfile);
        }
        if width == 0 || height == 0 {
            return Err(Reason::Malformed);
        }
        Ok(Self {
            decoder,
            input,
            width,
            height,
            channels,
        })
    }

    fn sample_count(&self) -> Option<usize> {
        self.width
            .checked_mul(self.height)?
            .checked_mul(self.channels)
    }

    fn storage_bytes(&self) -> Option<usize> {
        self.sample_count()?.checked_mul(size_of::<u16>())
    }

    fn decode(mut self) -> Result<Samples, Reason> {
        if advance(&mut self.decoder, &mut self.input, None)? != Event::FrameHeader {
            return Err(Reason::Malformed);
        }
        let count = self.sample_count().ok_or(Reason::TooLarge)?;
        let mut data = Vec::<u16>::new();
        data.try_reserve_exact(count)
            .map_err(|_| Reason::OutOfMemory)?;
        data.resize(count, 0);
        let mut buffers = [JxlOutputBuffer::new(
            bytemuck::cast_slice_mut(&mut data),
            self.height,
            self.width * self.channels * size_of::<u16>(),
        )];
        match advance(&mut self.decoder, &mut self.input, Some(&mut buffers))? {
            Event::FrameComplete { .. } => Ok(Samples {
                data,
                width: self.width,
                height: self.height,
                channels: self.channels,
            }),
            _ => Err(Reason::Malformed),
        }
    }
}

/// Maps base image pixel centers onto one axis of the gain map.
struct Axis {
    scale: f64,
    last: usize,
}

impl Axis {
    #[expect(clippy::cast_precision_loss)] // Image dimensions are far below 2^52.
    fn new(map: usize, base: u32) -> Self {
        Self {
            scale: map as f64 / f64::from(base),
            last: map - 1,
        }
    }

    /// The two samples around a base coordinate and the weight of the second.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )] // The position is clamped to the map before it is truncated.
    fn locate(&self, base: usize) -> (usize, usize, f32) {
        let position = ((base as f64 + 0.5) * self.scale - 0.5).clamp(0.0, self.last as f64);
        let lower = position as usize;
        (
            lower,
            (lower + 1).min(self.last),
            (position - lower as f64) as f32,
        )
    }
}

/// A decoded gain map, ready to reconstruct the alternate image.
pub(super) struct GainMap {
    samples: Samples,
    channels: [Channel; 3],
    space: Option<GainSpace>,
    base_width: usize,
    /// `None` when the gain map has the base image's dimensions.
    resampling: Option<(Axis, Axis)>,
    headroom_stops: f32,
}

impl GainMap {
    #[expect(clippy::cast_possible_truncation)] // The headroom is only reported.
    fn new(
        samples: Samples,
        metadata: &Metadata,
        space: Option<GainSpace>,
        base: &BaseImage,
    ) -> Self {
        let base_width = usize_from_u32(base.width);
        let base_height = usize_from_u32(base.height);
        let resampling =
            ((samples.width, samples.height) != (base_width, base_height)).then(|| {
                (
                    Axis::new(samples.width, base.width),
                    Axis::new(samples.height, base.height),
                )
            });
        Self {
            samples,
            channels: metadata.channels,
            space,
            base_width,
            resampling,
            headroom_stops: metadata.alternate_headroom as f32,
        }
    }

    /// Reconstructs the alternate image in place.
    ///
    /// `rgb` holds unpremultiplied linear BT.2020 pixels in reference-white
    /// units, in row-major order from pixel `first_pixel` of the base image.
    /// Finite input always gives finite output within the canonical range.
    pub(super) fn apply(&self, rgb: &mut [f32], first_pixel: usize) {
        for (offset, pixel) in rgb.chunks_exact_mut(3).enumerate() {
            let index = first_pixel + offset;
            let gains = self.gains(index % self.base_width, index / self.base_width);
            let mut value = [pixel[0], pixel[1], pixel[2]]
                .map(|value| value.clamp(-INTERMEDIATE_LIMIT, INTERMEDIATE_LIMIT));
            if let Some(space) = &self.space {
                value = multiply(&space.to_gain, value);
            }
            for (channel, parameters) in self.channels.iter().enumerate() {
                value[channel] = ((value[channel] + parameters.base_offset) * gains[channel]
                    - parameters.alternate_offset)
                    .clamp(-INTERMEDIATE_LIMIT, INTERMEDIATE_LIMIT);
            }
            // Gains do not commute with a color matrix, so map back afterwards.
            if let Some(space) = &self.space {
                value = multiply(&space.from_gain, value);
            }
            for (output, value) in pixel.iter_mut().zip(value) {
                *output = value.clamp(-MAX_OUTPUT, MAX_OUTPUT);
            }
        }
    }

    fn code(&self, x: usize, y: usize, channel: usize) -> u16 {
        let channel = channel.min(self.samples.channels - 1);
        self.samples.data[(y * self.samples.width + x) * self.samples.channels + channel]
    }

    /// Linear gains at one base image pixel.
    fn gains(&self, x: usize, y: usize) -> [f32; 3] {
        let Some((columns, rows)) = &self.resampling else {
            return std::array::from_fn(|channel| {
                self.gain(channel, f32::from(self.code(x, y, channel)))
            });
        };
        let (x0, x1, tx) = columns.locate(x);
        let (y0, y1, ty) = rows.locate(y);
        let mix = |a: u16, b: u16, weight: f32| {
            let (a, b) = (f32::from(a), f32::from(b));
            a + (b - a) * weight
        };
        std::array::from_fn(|channel| {
            let top = mix(self.code(x0, y0, channel), self.code(x1, y0, channel), tx);
            let bottom = mix(self.code(x0, y1, channel), self.code(x1, y1, channel), tx);
            self.gain(channel, top + (bottom - top) * ty)
        })
    }

    /// The gain for a sample code, which is fractional between gain map pixels.
    fn gain(&self, channel: usize, code: f32) -> f32 {
        self.channels[channel].multiplier(code / MAX_CODE)
    }
}

#[cfg(test)]
mod tests;
