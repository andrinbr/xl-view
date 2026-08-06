//! Binary-internal GPU facade. Stateful renderer ownership remains in
//! [`surface`], pure surface-selection and HDR/output policy live in [`output`],
//! and textual reports live in [`diagnostics`].

mod diagnostics;
mod hdr_metadata;
mod mip;
mod output;
mod presentation;
mod resampling;
mod surface;
mod tiles;
mod ui;
mod upload;
mod view;

pub use output::RenderingOptions;
use std::sync::Arc;
pub use surface::{GpuFailure, GpuState, initialize};
pub(crate) use surface::{GpuInitializationError, GpuOperationError};
pub(super) use ui::OverlaySection;

pub(super) type WorkReadyNotifier = Arc<dyn Fn() + Send + Sync>;

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

pub(super) fn create_fullscreen_render_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: None,
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(target_format.into())],
        }),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(super) const fn native_backends() -> wgpu::Backends {
    wgpu::Backends::VULKAN
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
const fn backend_unavailable_message() -> &'static str {
    "Vulkan is unavailable: the Vulkan loader, driver, or a compatible GPU adapter could not be found; install or update the GPU vendor's Vulkan-capable driver"
}

#[cfg(target_os = "macos")]
pub(super) const fn native_backends() -> wgpu::Backends {
    wgpu::Backends::METAL
}

#[cfg(target_os = "macos")]
const fn backend_unavailable_message() -> &'static str {
    "Metal is unavailable: no compatible GPU adapter could be found; update macOS and verify that this Mac supports Metal"
}
