//! Binary-internal GPU facade. Stateful renderer ownership remains in
//! [`surface`], pure surface-selection and HDR/output policy live in [`output`],
//! and textual reports live in [`diagnostics`].

mod diagnostics;
mod display_target;
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
pub(crate) use surface::{
    DisplayRefresh, GpuInitializationError, GpuOperationError, SurfaceCapabilities,
};
pub use surface::{GpuFailure, GpuState};
pub(super) use ui::OverlaySection;
use winit::window::Window;

use crate::cli::OutputMode;

pub(super) type FailureReporter = Arc<dyn Fn(GpuFailure) + Send + Sync>;
pub(super) type DisplayChangeNotifier = Arc<dyn Fn() + Send + Sync>;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeBackend {
    name: &'static str,
    backends: wgpu::Backends,
    unavailable_message: &'static str,
}

impl NativeBackend {
    pub(super) const fn name(self) -> &'static str {
        self.name
    }

    pub(super) const fn backends(self) -> wgpu::Backends {
        self.backends
    }

    pub(super) const fn unavailable_message(self) -> &'static str {
        self.unavailable_message
    }
}

#[cfg(target_os = "linux")]
const NATIVE_BACKEND_ORDER: &[NativeBackend] = &[NativeBackend {
    name: "Vulkan",
    backends: wgpu::Backends::VULKAN,
    unavailable_message: "Vulkan is unavailable: the Vulkan loader, driver, or a compatible GPU adapter could not be found. Install or update the GPU vendor's Vulkan-capable driver",
}];

#[cfg(target_os = "windows")]
const NATIVE_BACKEND_ORDER: &[NativeBackend] = &[
    NativeBackend {
        name: "Vulkan",
        backends: wgpu::Backends::VULKAN,
        unavailable_message: "Vulkan is unavailable: the Vulkan loader, driver, or a compatible GPU adapter could not be found. Install or update the GPU vendor's Vulkan-capable driver",
    },
    NativeBackend {
        name: "DirectX 12",
        backends: wgpu::Backends::DX12,
        unavailable_message: "Vulkan and DirectX 12 are unavailable: neither backend found a compatible GPU adapter. Vulkan was tried first, followed by DirectX 12. Install or update the GPU driver and Windows",
    },
];

#[cfg(target_os = "macos")]
const NATIVE_BACKEND_ORDER: &[NativeBackend] = &[NativeBackend {
    name: "Metal",
    backends: wgpu::Backends::METAL,
    unavailable_message: "Metal is unavailable: no compatible GPU adapter could be found. Update macOS and verify that this Mac supports Metal",
}];

pub(super) const fn native_backend_order() -> &'static [NativeBackend] {
    NATIVE_BACKEND_ORDER
}

#[cfg(test)]
pub(super) const fn native_backends() -> wgpu::Backends {
    let mut backends = wgpu::Backends::empty();
    let mut index = 0;
    while index < NATIVE_BACKEND_ORDER.len() {
        backends = backends.union(NATIVE_BACKEND_ORDER[index].backends());
        index += 1;
    }
    backends
}

#[allow(clippy::too_many_arguments)] // Keeps initialization dependencies explicit across the backend retry boundary.
pub(super) fn initialize(
    mut descriptor: impl FnMut() -> wgpu::InstanceDescriptor,
    window: &Arc<dyn Window>,
    output_mode: OutputMode,
    rendering_options: RenderingOptions,
    diagnostics_pattern: bool,
    report_failure: &FailureReporter,
    notify_work_ready: &WorkReadyNotifier,
    notify_display_changed: &DisplayChangeNotifier,
    gpu_memory_limit_bytes: u64,
) -> Result<GpuState, GpuInitializationError> {
    for (index, backend) in native_backend_order().iter().copied().enumerate() {
        let mut descriptor = descriptor();
        descriptor.backends = backend.backends();
        let instance = wgpu::Instance::new(descriptor);
        match surface::initialize(
            &instance,
            backend,
            Arc::clone(window),
            output_mode,
            rendering_options,
            diagnostics_pattern,
            Arc::clone(report_failure),
            Arc::clone(notify_work_ready),
            Arc::clone(notify_display_changed),
            gpu_memory_limit_bytes,
        ) {
            Ok(gpu) => return Ok(gpu),
            Err(
                error @ (GpuInitializationError::BackendUnavailable { .. }
                | GpuInitializationError::SurfaceCreation(_)),
            ) => {
                if let Some(fallback) = native_backend_order().get(index + 1) {
                    tracing::warn!(
                        backend = backend.name(),
                        fallback_backend = fallback.name(),
                        error = %error,
                        "GPU backend initialization failed, trying fallback"
                    );
                } else {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!("at least one native GPU backend must be attempted")
}

#[cfg(all(test, target_os = "windows"))]
mod tests {
    use super::native_backend_order;

    #[test]
    fn windows_tries_vulkan_before_directx_12() {
        let order = native_backend_order();

        assert_eq!(order.len(), 2);
        assert_eq!(order[0].name(), "Vulkan");
        assert_eq!(order[0].backends(), wgpu::Backends::VULKAN);
        assert_eq!(order[1].name(), "DirectX 12");
        assert_eq!(order[1].backends(), wgpu::Backends::DX12);
    }
}
