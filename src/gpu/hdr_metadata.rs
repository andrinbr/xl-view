//! Platform adapter for optional HDR metadata signaling.

#[cfg(not(target_vendor = "apple"))]
pub(super) use vulkan_hdr_metadata::{HdrMetadata, HdrSurface, SignalStatus, request_device};

#[cfg(target_vendor = "apple")]
mod apple {
    use std::fmt;

    /// HDR luminance values retained for future `CAEDRMetadata` integration.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct HdrMetadata {
        _mastering_max: f32,
        _mastering_min: f32,
        _max_cll: f32,
        _max_fall: f32,
    }

    impl HdrMetadata {
        pub fn bt2020(
            mastering_max_luminance_nits: f32,
            mastering_min_luminance_nits: f32,
            max_content_light_level_nits: f32,
            max_frame_average_light_level_nits: f32,
        ) -> Self {
            Self {
                _mastering_max: mastering_max_luminance_nits,
                _mastering_min: mastering_min_luminance_nits,
                _max_cll: max_content_light_level_nits,
                _max_fall: max_frame_average_light_level_nits,
            }
        }
    }

    /// Result of attempting to associate HDR metadata with a Metal surface.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum SignalStatus {
        NotRequested,
        Unsupported,
    }

    impl fmt::Display for SignalStatus {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(match self {
                Self::NotRequested => "HDR metadata not requested",
                Self::Unsupported => "HDR metadata signaling unsupported",
            })
        }
    }

    #[derive(Debug)]
    pub struct HdrMetadataSignaler {
        device: wgpu::Device,
        supported: bool,
    }

    impl HdrMetadataSignaler {
        pub const fn is_supported(&self) -> bool {
            self.supported
        }

        pub fn bind_surface(self, surface: wgpu::Surface<'_>) -> HdrSurface<'_> {
            HdrSurface {
                surface,
                device: self.device,
            }
        }
    }

    #[derive(Debug)]
    pub struct HdrSurface<'window> {
        surface: wgpu::Surface<'window>,
        device: wgpu::Device,
    }

    impl<'window> HdrSurface<'window> {
        pub fn get_capabilities(&self, adapter: &wgpu::Adapter) -> wgpu::SurfaceCapabilities {
            self.surface.get_capabilities(adapter)
        }

        pub fn display_hdr_info(&self, adapter: &wgpu::Adapter) -> wgpu::DisplayHdrInfo {
            self.surface.display_hdr_info(adapter)
        }

        pub fn get_current_texture(&self) -> wgpu::CurrentSurfaceTexture {
            self.surface.get_current_texture()
        }

        pub fn replace(&mut self, surface: wgpu::Surface<'window>) {
            self.surface = surface;
        }

        pub fn configure(
            &mut self,
            config: &wgpu::SurfaceConfiguration,
            metadata: Option<HdrMetadata>,
        ) -> SignalStatus {
            self.surface.configure(&self.device, config);
            Self::metadata_status(metadata)
        }

        #[allow(clippy::unused_self)] // Mirrors the platform surface interface used by the renderer.
        pub fn set_metadata(&self, metadata: Option<HdrMetadata>) -> SignalStatus {
            Self::metadata_status(metadata)
        }

        fn metadata_status(metadata: Option<HdrMetadata>) -> SignalStatus {
            match metadata {
                Some(_) => SignalStatus::Unsupported,
                None => SignalStatus::NotRequested,
            }
        }
    }

    pub async fn request_device(
        adapter: &wgpu::Adapter,
        descriptor: &wgpu::DeviceDescriptor<'_>,
    ) -> Result<(wgpu::Device, wgpu::Queue, HdrMetadataSignaler), wgpu::RequestDeviceError> {
        let (device, queue) = adapter.request_device(descriptor).await?;
        let signaler = HdrMetadataSignaler {
            device: device.clone(),
            supported: false,
        };
        Ok((device, queue, signaler))
    }
}

#[cfg(target_vendor = "apple")]
pub(super) use apple::{HdrMetadata, HdrSurface, SignalStatus, request_device};
