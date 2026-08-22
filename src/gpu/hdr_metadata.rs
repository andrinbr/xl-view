//! Platform adapter for Linux's optional HDR metadata bridge.
//!
//! Windows and macOS consume XL-View's already mapped output through their PQ
//! surface color space, so requesting another tone-mapping hint is unnecessary.

#[cfg(target_os = "linux")]
pub(super) use vulkan_hdr_metadata::{HdrMetadata, HdrSurface, SignalStatus, request_device};

#[cfg(not(target_os = "linux"))]
// Mirror the Linux bridge API so the renderer has one surface lifecycle.
// This adapter delegates to wgpu without emitting HDR metadata.
mod passthrough {
    use std::fmt;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct HdrMetadata;

    impl HdrMetadata {
        pub const fn display_mapped_bt2020(_peak_nits: f32) -> Self {
            Self
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum SignalStatus {
        NotRequested,
        NotRequired,
    }

    impl fmt::Display for SignalStatus {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(match self {
                Self::NotRequested => "HDR metadata not requested",
                Self::NotRequired => "HDR metadata not required on this platform",
            })
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

        pub fn get_current_texture(&mut self) -> wgpu::CurrentSurfaceTexture {
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

        #[allow(clippy::unused_self)]
        pub fn set_metadata(&mut self, metadata: Option<HdrMetadata>) -> SignalStatus {
            Self::metadata_status(metadata)
        }

        const fn metadata_status(metadata: Option<HdrMetadata>) -> SignalStatus {
            match metadata {
                Some(_) => SignalStatus::NotRequired,
                None => SignalStatus::NotRequested,
            }
        }
    }

    pub async fn request_device<'window>(
        adapter: &wgpu::Adapter,
        descriptor: &wgpu::DeviceDescriptor<'_>,
        surface: wgpu::Surface<'window>,
    ) -> Result<(wgpu::Device, wgpu::Queue, HdrSurface<'window>), wgpu::RequestDeviceError> {
        let (device, queue) = adapter.request_device(descriptor).await?;
        let hdr_surface = HdrSurface {
            surface,
            device: device.clone(),
        };
        Ok((device, queue, hdr_surface))
    }
}

#[cfg(not(target_os = "linux"))]
pub(super) use passthrough::{HdrMetadata, HdrSurface, SignalStatus, request_device};
