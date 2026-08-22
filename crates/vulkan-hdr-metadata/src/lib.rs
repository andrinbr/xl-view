#![cfg(target_os = "linux")]

//! Minimal Linux Vulkan bridge for attaching HDR10 static metadata to a wgpu
//! swapchain. It bounds the already display-mapped BT.2020/PQ signal so a
//! Wayland compositor does not treat it as unconstrained 10,000-nit content.

use std::fmt;

use ash::vk;

const BT2020_RED: vk::XYColorEXT = vk::XYColorEXT { x: 0.708, y: 0.292 };
const BT2020_GREEN: vk::XYColorEXT = vk::XYColorEXT { x: 0.170, y: 0.797 };
const BT2020_BLUE: vk::XYColorEXT = vk::XYColorEXT { x: 0.131, y: 0.046 };
const D65_WHITE: vk::XYColorEXT = vk::XYColorEXT {
    x: 0.3127,
    y: 0.3290,
};

/// Static metadata for XL-View's already display-mapped BT.2020/PQ output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrMetadata {
    peak_nits: f32,
}

impl HdrMetadata {
    /// Describes output bounded to `peak_nits` by the presentation shader.
    #[must_use]
    pub fn display_mapped_bt2020(peak_nits: f32) -> Self {
        Self {
            peak_nits: known_non_negative(peak_nits),
        }
    }

    fn to_vulkan(self) -> vk::HdrMetadataEXT<'static> {
        vk::HdrMetadataEXT::default()
            .display_primary_red(BT2020_RED)
            .display_primary_green(BT2020_GREEN)
            .display_primary_blue(BT2020_BLUE)
            .white_point(D65_WHITE)
            .max_luminance(self.peak_nits)
            .min_luminance(0.0)
            .max_content_light_level(self.peak_nits)
            .max_frame_average_light_level(0.0)
    }
}

fn known_non_negative(value: f32) -> f32 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}

/// Result of attempting to associate HDR metadata with a swapchain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignalStatus {
    Signaled,
    NotRequested,
    Unsupported,
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

/// A wgpu surface bound to the device used by the metadata extension.
#[derive(Debug)]
pub struct HdrSurface<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    metadata_supported: bool,
    configured: bool,
}

impl<'window> HdrSurface<'window> {
    #[must_use]
    pub fn get_capabilities(&self, adapter: &wgpu::Adapter) -> wgpu::SurfaceCapabilities {
        self.surface.get_capabilities(adapter)
    }

    #[must_use]
    pub fn display_hdr_info(&self, adapter: &wgpu::Adapter) -> wgpu::DisplayHdrInfo {
        self.surface.display_hdr_info(adapter)
    }

    #[must_use]
    pub fn get_current_texture(&mut self) -> wgpu::CurrentSurfaceTexture {
        self.surface.get_current_texture()
    }

    pub fn replace(&mut self, surface: wgpu::Surface<'window>) {
        self.surface = surface;
        self.configured = false;
    }

    /// Configures the swapchain and immediately describes its mapped output.
    pub fn configure(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        metadata: Option<HdrMetadata>,
    ) -> SignalStatus {
        self.surface.configure(&self.device, config);
        self.configured = true;
        self.set_metadata(metadata)
    }

    /// Updates metadata on an existing swapchain after its output target changes.
    pub fn set_metadata(&mut self, metadata: Option<HdrMetadata>) -> SignalStatus {
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

    #[allow(unsafe_code)]
    fn set_vulkan_metadata(&self, metadata: HdrMetadata) -> SignalStatus {
        // SAFETY: The guards remain alive for the complete raw Vulkan call.
        let Some(hal_surface) = (unsafe { self.surface.as_hal::<wgpu::hal::api::Vulkan>() }) else {
            return SignalStatus::SurfaceUnavailable;
        };
        // SAFETY: `self.device` owns the returned HAL device guard.
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
        // SAFETY: The extension is enabled, both handles outlive this call, and
        // the mutable bridge borrow keeps its surface operations sequential.
        unsafe {
            extension.set_hdr_metadata(&[swapchain], &[vk_metadata]);
        }
        SignalStatus::Signaled
    }
}

/// Requests a wgpu device while opportunistically enabling `VK_EXT_hdr_metadata`.
/// Ordinary device creation remains the fallback when it is unavailable.
///
/// # Errors
///
/// Returns an error when ordinary device creation fails after the optional
/// extension path cannot be used.
pub async fn request_device<'window>(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
    surface: wgpu::Surface<'window>,
) -> Result<(wgpu::Device, wgpu::Queue, HdrSurface<'window>), wgpu::RequestDeviceError> {
    if let Some((device, queue)) = try_request_vulkan_device(adapter, descriptor) {
        let hdr_surface = HdrSurface {
            surface,
            device: device.clone(),
            metadata_supported: true,
            configured: false,
        };
        return Ok((device, queue, hdr_surface));
    }

    // HDR metadata is optional. The bound surface reports it as unsupported
    // when the Vulkan extension path cannot be used.
    let (device, queue) = adapter.request_device(descriptor).await?;
    let hdr_surface = HdrSurface {
        surface,
        device: device.clone(),
        metadata_supported: false,
        configured: false,
    };
    Ok((device, queue, hdr_surface))
}

#[allow(unsafe_code)]
fn try_request_vulkan_device(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    // SAFETY: The guard is used only for capability inspection and device creation.
    let hal_adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    if !hal_adapter
        .physical_device_capabilities()
        .supports_extension(ash::ext::hdr_metadata::NAME)
    {
        return None;
    }

    let callback: Box<wgpu::hal::vulkan::CreateDeviceCallback<'_>> = Box::new(|arguments| {
        if !arguments.extensions.contains(&ash::ext::hdr_metadata::NAME) {
            arguments.extensions.push(ash::ext::hdr_metadata::NAME);
        }
    });
    // SAFETY: The callback enables only the extension queried above.
    let hal_device = unsafe {
        hal_adapter.open_with_callback(
            descriptor.required_features,
            &descriptor.required_limits,
            &descriptor.memory_hints,
            Some(callback),
        )
    }
    .ok()?;
    drop(hal_adapter);

    // SAFETY: The HAL device belongs to this adapter and matches `descriptor`.
    unsafe { adapter.create_device_from_hal(hal_device, descriptor) }.ok()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn mapped_output_metadata_uses_peak_for_mastering_and_content_bounds() {
        let metadata = HdrMetadata::display_mapped_bt2020(1_000.0).to_vulkan();

        assert_eq!(metadata.display_primary_red.x, 0.708);
        assert_eq!(metadata.display_primary_green.y, 0.797);
        assert_eq!(metadata.display_primary_blue.y, 0.046);
        assert_eq!(metadata.white_point.x, 0.3127);
        assert_eq!(metadata.max_luminance, 1_000.0);
        assert_eq!(metadata.min_luminance, 0.0);
        assert_eq!(metadata.max_content_light_level, 1_000.0);
        assert_eq!(metadata.max_frame_average_light_level, 0.0);
    }

    #[test]
    fn mapped_output_metadata_sanitizes_invalid_peaks() {
        assert_eq!(HdrMetadata::display_mapped_bt2020(f32::NAN).peak_nits, 0.0);
        assert_eq!(HdrMetadata::display_mapped_bt2020(-1.0).peak_nits, 0.0);
    }
}
