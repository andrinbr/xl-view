#![cfg(not(target_vendor = "apple"))]

//! wgpu supports PQ/BT.2100 swapchains, but currently has no public API for
//! submitting HDR10 static metadata. On Vulkan, we therefore use
//! `VK_EXT_hdr_metadata` as a small backend-specific bridge. Metadata should be
//! set after swapchain configuration and before the presentation that should
//! use it. If the extension is unavailable, PQ output continues without
//! explicit metadata.
//!
//! Native HDR metadata support was mentioned as possible follow-up work in:
//! <https://github.com/gfx-rs/wgpu/pull/9658>

use std::error::Error;
use std::fmt;
use std::sync::Mutex;

use ash::vk;

const BT2020_RED: Chromaticity = Chromaticity::new(0.708, 0.292);
const BT2020_GREEN: Chromaticity = Chromaticity::new(0.170, 0.797);
const BT2020_BLUE: Chromaticity = Chromaticity::new(0.131, 0.046);
const D65_WHITE: Chromaticity = Chromaticity::new(0.3127, 0.3290);

/// CIE 1931 xy chromaticity coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Chromaticity {
    x: f32,
    y: f32,
}

impl Chromaticity {
    const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn to_vulkan(self) -> vk::XYColorEXT {
        vk::XYColorEXT {
            x: self.x,
            y: self.y,
        }
    }
}

/// SMPTE ST 2086 and CTA-861.3 metadata associated with a BT.2020 surface.
///
/// A zero luminance value means unknown, as specified by
/// `VkHdrMetadataEXT`. Invalid or negative inputs are converted to zero before
/// reaching Vulkan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrMetadata {
    mastering_max: f32,
    mastering_min: f32,
    max_cll: f32,
    max_fall: f32,
}

impl HdrMetadata {
    /// Creates metadata for content represented in xl-view's synthesized
    /// BT.2020/D65 output volume. These primaries describe the viewer's output
    /// assumption, not necessarily the source content's mastering display.
    #[must_use]
    pub fn bt2020(
        mastering_max_luminance_nits: f32,
        mastering_min_luminance_nits: f32,
        max_content_light_level_nits: f32,
        max_frame_average_light_level_nits: f32,
    ) -> Self {
        let mastering_max_luminance_nits = known_non_negative(mastering_max_luminance_nits);
        let mastering_min_luminance_nits =
            clamp_to_known_upper_bound(mastering_min_luminance_nits, mastering_max_luminance_nits);
        let max_content_light_level_nits = known_non_negative(max_content_light_level_nits);
        let max_frame_average_light_level_nits = clamp_to_known_upper_bound(
            max_frame_average_light_level_nits,
            max_content_light_level_nits,
        );
        Self {
            mastering_max: mastering_max_luminance_nits,
            mastering_min: mastering_min_luminance_nits,
            max_cll: max_content_light_level_nits,
            max_fall: max_frame_average_light_level_nits,
        }
    }

    fn to_vulkan(self) -> vk::HdrMetadataEXT<'static> {
        vk::HdrMetadataEXT::default()
            .display_primary_red(BT2020_RED.to_vulkan())
            .display_primary_green(BT2020_GREEN.to_vulkan())
            .display_primary_blue(BT2020_BLUE.to_vulkan())
            .white_point(D65_WHITE.to_vulkan())
            .max_luminance(self.mastering_max)
            .min_luminance(self.mastering_min)
            .max_content_light_level(self.max_cll)
            .max_frame_average_light_level(self.max_fall)
    }
}

fn known_non_negative(value: f32) -> f32 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}

fn clamp_to_known_upper_bound(value: f32, upper_bound: f32) -> f32 {
    let value = known_non_negative(value);
    if upper_bound == 0.0 {
        value
    } else {
        value.min(upper_bound)
    }
}

/// Result of attempting to associate HDR metadata with a swapchain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignalStatus {
    /// Metadata was passed to Vulkan and will apply on the next presentation.
    Signaled,
    /// The caller did not request a metadata submission.
    NotRequested,
    /// The Vulkan device does not expose `VK_EXT_hdr_metadata`.
    Unsupported,
    /// The surface has no native Vulkan swapchain to which metadata can be attached.
    SurfaceUnavailable,
}

impl fmt::Display for SignalStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Signaled => "signaled through VK_EXT_hdr_metadata",
            Self::NotRequested => "HDR metadata not requested",
            Self::Unsupported => "VK_EXT_hdr_metadata unavailable; compositor fallback",
            Self::SurfaceUnavailable => "native Vulkan swapchain unavailable; compositor fallback",
        })
    }
}

/// A `wgpu` device request failure.
#[derive(Debug)]
pub enum RequestDeviceError {
    Wgpu(wgpu::RequestDeviceError),
    Vulkan(wgpu::hal::DeviceError),
}

impl fmt::Display for RequestDeviceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wgpu(error) => error.fmt(formatter),
            Self::Vulkan(error) => write!(formatter, "Vulkan device creation failed: {error}"),
        }
    }
}

impl Error for RequestDeviceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Wgpu(error) => Some(error),
            Self::Vulkan(error) => Some(error),
        }
    }
}

/// Device-bound access to optional Vulkan HDR metadata signaling.
///
/// The stored device clone guarantees that metadata is submitted with the
/// logical device used to create the swapchain. Calls are serialized because
/// Vulkan requires external synchronization for each swapchain passed to
/// `vkSetHdrMetadataEXT`.
#[derive(Debug)]
pub struct HdrMetadataSignaler {
    device: wgpu::Device,
    supported: bool,
}

impl HdrMetadataSignaler {
    /// Whether the selected Vulkan device enabled `VK_EXT_hdr_metadata`.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.supported
    }

    /// Consumes the device token and surface, preventing callers from
    /// configuring the surface with a different device behind the metadata
    /// bridge.
    pub fn bind_surface(self, surface: wgpu::Surface<'_>) -> HdrSurface<'_> {
        HdrSurface {
            surface,
            device: self.device,
            metadata_supported: self.supported,
            configured: false,
            signal_lock: Mutex::new(()),
        }
    }
}

/// A presentable `wgpu` surface bound to the logical device used for HDR
/// metadata signaling.
///
/// The raw `wgpu::Surface` is deliberately not exposed. This makes it
/// impossible for safe callers to reconfigure its Vulkan swapchain with a
/// different device before a metadata update.
#[derive(Debug)]
pub struct HdrSurface<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    metadata_supported: bool,
    configured: bool,
    signal_lock: Mutex<()>,
}

impl<'window> HdrSurface<'window> {
    /// Whether the selected Vulkan device enabled `VK_EXT_hdr_metadata`.
    #[must_use]
    pub fn is_metadata_supported(&self) -> bool {
        self.metadata_supported
    }

    /// Returns the capabilities reported by the wrapped surface.
    #[must_use]
    pub fn get_capabilities(&self, adapter: &wgpu::Adapter) -> wgpu::SurfaceCapabilities {
        self.surface.get_capabilities(adapter)
    }

    /// Returns the display information reported by the wrapped surface.
    #[must_use]
    pub fn display_hdr_info(&self, adapter: &wgpu::Adapter) -> wgpu::DisplayHdrInfo {
        self.surface.display_hdr_info(adapter)
    }

    /// Acquires the next presentable texture from the wrapped surface.
    ///
    /// Acquisition and metadata submission are serialized because both host
    /// operations access the Vulkan swapchain.
    #[must_use]
    pub fn get_current_texture(&self) -> wgpu::CurrentSurfaceTexture {
        let _serial = self
            .signal_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.surface.get_current_texture()
    }

    /// Replaces a lost surface. The new surface remains unconfigured until the
    /// next call to [`Self::configure`].
    pub fn replace(&mut self, surface: wgpu::Surface<'window>) {
        self.surface = surface;
        self.configured = false;
    }

    /// Configures the swapchain with the bound device, then attaches optional
    /// HDR metadata.
    pub fn configure(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        metadata: Option<HdrMetadata>,
    ) -> SignalStatus {
        self.surface.configure(&self.device, config);
        self.configured = true;
        self.set_metadata(metadata)
    }

    /// Updates metadata on an already-configured swapchain.
    ///
    /// `None` leaves any previously submitted Vulkan metadata unchanged. A new
    /// swapchain receives no metadata until `Some` is submitted for it.
    pub fn set_metadata(&self, metadata: Option<HdrMetadata>) -> SignalStatus {
        let Some(metadata) = metadata else {
            return SignalStatus::NotRequested;
        };
        if !self.metadata_supported {
            return SignalStatus::Unsupported;
        }
        if !self.configured {
            return SignalStatus::SurfaceUnavailable;
        }
        self.set_vulkan_metadata(metadata)
    }

    /// The raw handles are borrowed from guards and used only for the duration
    /// of the Vulkan call. The device was created from the same adapter and is
    /// retained by `self`; the raw surface is private and all configuration
    /// goes through `configure`, which establishes the required
    /// device/swapchain relationship. The extension is checked at device
    /// creation and again before loading its function pointer.
    #[allow(unsafe_code)]
    fn set_vulkan_metadata(&self, metadata: HdrMetadata) -> SignalStatus {
        let _serial = self
            .signal_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: The guards and their resources are neither destroyed nor
        // mutated. They remain alive through the raw Vulkan call below.
        let Some(hal_surface) = (unsafe { self.surface.as_hal::<wgpu::hal::api::Vulkan>() }) else {
            return SignalStatus::SurfaceUnavailable;
        };
        // SAFETY: As above; `self.device` owns the returned HAL device.
        let Some(hal_device) = (unsafe { self.device.as_hal::<wgpu::hal::api::Vulkan>() }) else {
            return SignalStatus::SurfaceUnavailable;
        };
        if !hal_device
            .enabled_device_extensions()
            .contains(&ash::ext::hdr_metadata::NAME)
        {
            return SignalStatus::Unsupported;
        }
        let Some(swapchain) = hal_surface.raw_native_swapchain() else {
            return SignalStatus::SurfaceUnavailable;
        };
        let extension = ash::ext::hdr_metadata::Device::new(
            hal_device.shared_instance().raw_instance(),
            hal_device.raw_device(),
        );
        let vk_metadata = metadata.to_vulkan();
        // SAFETY: The extension is enabled, `swapchain` was configured with
        // `self.device`, both handles outlive this call, the slices have equal
        // non-zero length, and `signal_lock` serializes metadata writes.
        unsafe {
            extension.set_hdr_metadata(&[swapchain], &[vk_metadata]);
        }
        SignalStatus::Signaled
    }
}

/// Requests a normal `wgpu` device, enabling `VK_EXT_hdr_metadata` as an
/// optional Vulkan device extension when the physical device advertises it.
///
/// Unsupported backends and Vulkan devices use `Adapter::request_device`
/// unchanged. HDR rendering therefore never depends on metadata support.
///
/// # Errors
///
/// Returns an error if ordinary `wgpu` device creation fails. Failure of the
/// optional metadata-enabled Vulkan attempt is logged and retried without the
/// extension.
pub async fn request_device(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> Result<(wgpu::Device, wgpu::Queue, HdrMetadataSignaler), RequestDeviceError> {
    match try_request_vulkan_device(adapter, descriptor) {
        Ok(Some(result)) => return Ok(result),
        Ok(None) => {}
        Err(error) => {
            tracing::warn!(
                %error,
                extension = "VK_EXT_hdr_metadata",
                fallback = "ordinary wgpu device",
                "cannot enable optional Vulkan HDR metadata; retrying without it"
            );
        }
    }

    let (device, queue) = adapter
        .request_device(descriptor)
        .await
        .map_err(RequestDeviceError::Wgpu)?;
    let signaler = HdrMetadataSignaler {
        device: device.clone(),
        supported: false,
    };
    Ok((device, queue, signaler))
}

/// The HAL adapter guard is read-only. The callback adds only an extension
/// reported by the physical device. The resulting HAL device is immediately
/// returned to the same wgpu adapter with exactly the descriptor features used
/// to create it.
#[allow(unsafe_code)]
fn try_request_vulkan_device(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> Result<Option<(wgpu::Device, wgpu::Queue, HdrMetadataSignaler)>, RequestDeviceError> {
    // SAFETY: The guard is used only for capability inspection and device
    // creation. No HAL resource is destroyed or retained past the guard.
    let Some(hal_adapter) = (unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }) else {
        return Ok(None);
    };
    if !hal_adapter
        .physical_device_capabilities()
        .supports_extension(ash::ext::hdr_metadata::NAME)
    {
        return Ok(None);
    }

    let callback: Box<wgpu::hal::vulkan::CreateDeviceCallback<'_>> = Box::new(|arguments| {
        if !arguments.extensions.contains(&ash::ext::hdr_metadata::NAME) {
            arguments.extensions.push(ash::ext::hdr_metadata::NAME);
        }
    });
    // SAFETY: The callback only enables a device extension whose support was
    // queried above, satisfying `open_with_callback`'s contract.
    let hal_device = unsafe {
        hal_adapter.open_with_callback(
            descriptor.required_features,
            &descriptor.required_limits,
            &descriptor.memory_hints,
            Some(callback),
        )
    }
    .map_err(RequestDeviceError::Vulkan)?;
    drop(hal_adapter);

    // SAFETY: `hal_device` came from this adapter, and was opened with the same
    // requested features and limits as `descriptor`.
    let (device, queue) = unsafe { adapter.create_device_from_hal(hal_device, descriptor) }
        .map_err(RequestDeviceError::Wgpu)?;
    let signaler = HdrMetadataSignaler {
        device: device.clone(),
        supported: true,
    };
    Ok(Some((device, queue, signaler)))
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Exact constants verify lossless Vulkan field construction.
mod tests {
    use super::*;

    #[test]
    fn metadata_sanitizes_unknown_and_impossible_values() {
        let metadata = HdrMetadata::bt2020(f32::NAN, -1.0, 1_000.0, 2_000.0);
        assert_eq!(metadata.mastering_max, 0.0);
        assert_eq!(metadata.mastering_min, 0.0);
        assert_eq!(metadata.max_cll, 1_000.0);
        assert_eq!(metadata.max_fall, 1_000.0);
    }

    #[test]
    fn metadata_preserves_values_with_unknown_upper_bounds() {
        let metadata = HdrMetadata::bt2020(0.0, 0.005, 0.0, 300.0);
        assert_eq!(metadata.mastering_max, 0.0);
        assert_eq!(metadata.mastering_min, 0.005);
        assert_eq!(metadata.max_cll, 0.0);
        assert_eq!(metadata.max_fall, 300.0);
    }

    #[test]
    fn bt2020_vulkan_metadata_uses_standard_primaries() {
        let metadata = HdrMetadata::bt2020(1_000.0, 0.005, 800.0, 300.0).to_vulkan();
        assert_eq!(metadata.display_primary_red.x, 0.708);
        assert_eq!(metadata.display_primary_green.y, 0.797);
        assert_eq!(metadata.display_primary_blue.y, 0.046);
        assert_eq!(metadata.white_point.x, 0.3127);
        assert_eq!(metadata.max_luminance, 1_000.0);
        assert_eq!(metadata.min_luminance, 0.005);
        assert_eq!(metadata.max_content_light_level, 800.0);
        assert_eq!(metadata.max_frame_average_light_level, 300.0);
    }
}
