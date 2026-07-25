use std::io;

use thiserror::Error;

use crate::gpu::{GpuFailure, GpuInitializationError, GpuOperationError};

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("cannot start decode coordinator: {0}")]
    DecodeCoordinator(#[source] io::Error),

    #[error("failed to create the application window: {0}")]
    WindowCreation(#[source] winit::error::RequestError),

    #[error(transparent)]
    Gpu(#[from] GpuFailure),

    #[error("failed to prepare image rendering data: {0}")]
    GpuBackgroundWork(#[source] io::Error),

    #[error("failed to install decoded image on the GPU: {0}")]
    ImageInstall(#[source] GpuOperationError),

    #[error(transparent)]
    GpuInitialization(#[from] GpuInitializationError),

    #[error("failed to reconfigure the surface after resize: {0}")]
    SurfaceResize(#[source] GpuOperationError),

    #[error("failed to reconfigure the surface after scale-factor change: {0}")]
    SurfaceScaleFactor(#[source] GpuOperationError),

    #[error("failed to refresh the surface after moving outputs: {0}")]
    DisplayMove(#[source] GpuOperationError),

    #[error("failed to refresh the surface after a possible display change: {0}")]
    DisplayFocus(#[source] GpuOperationError),

    #[error("failed to render the frame: {0}")]
    Render(#[source] GpuOperationError),
}

impl RuntimeError {
    pub const fn category(&self) -> &'static str {
        match self {
            Self::DecodeCoordinator(_) => "decode_coordinator",
            Self::WindowCreation(_) => "window_creation",
            Self::Gpu(failure) => failure.category(),
            Self::GpuBackgroundWork(_) => "gpu_background_work",
            Self::ImageInstall(_) => "image_install",
            Self::GpuInitialization(error) => error.category(),
            Self::SurfaceResize(_) => "surface_resize",
            Self::SurfaceScaleFactor(_) => "surface_scale_factor",
            Self::DisplayMove(_) => "display_move",
            Self::DisplayFocus(_) => "display_focus",
            Self::Render(_) => "render",
        }
    }
}

#[derive(Debug, Error)]
pub enum AppError {
    #[cfg(target_os = "linux")]
    #[error(
        "xl-view requires a native Wayland session; neither WAYLAND_DISPLAY nor WAYLAND_SOCKET is set"
    )]
    NoNativeWayland,

    #[error("failed to create the application event loop: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),

    #[error(transparent)]
    Runtime(#[from] RuntimeError),
}

impl AppError {
    pub const fn category(&self) -> &'static str {
        match self {
            #[cfg(target_os = "linux")]
            Self::NoNativeWayland => "no_native_wayland",
            Self::EventLoop(_) => "event_loop",
            Self::Runtime(error) => error.category(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn runtime_errors_preserve_category_and_source() {
        let error = AppError::from(RuntimeError::GpuBackgroundWork(io::Error::other(
            "tile worker disconnected",
        )));

        assert_eq!(error.category(), "gpu_background_work");
        let AppError::Runtime(RuntimeError::GpuBackgroundWork(runtime_source)) = &error else {
            panic!("runtime error lost its typed variant");
        };
        assert_eq!(runtime_source.to_string(), "tile worker disconnected");
        assert_eq!(
            error.to_string(),
            "failed to prepare image rendering data: tile worker disconnected"
        );
        assert_eq!(
            error.source().unwrap().to_string(),
            "tile worker disconnected"
        );
    }

    #[test]
    fn backend_discovery_failure_has_its_own_runtime_category() {
        let initialization =
            GpuInitializationError::backend_unavailable(wgpu::RequestAdapterError::EnvNotSet);
        let error = AppError::from(RuntimeError::from(initialization));

        assert_eq!(error.category(), "gpu_backend_unavailable");
        assert!(error.to_string().contains("is unavailable"));
    }

    #[test]
    fn gpu_operation_errors_preserve_runtime_context_and_typed_sources() {
        let error = RuntimeError::ImageInstall(GpuOperationError::from(io::Error::other(
            "coarse texture allocation failed",
        )));

        assert_eq!(error.category(), "image_install");
        assert_eq!(
            error.to_string(),
            "failed to install decoded image on the GPU: coarse texture allocation failed"
        );
        let RuntimeError::ImageInstall(GpuOperationError::ImageResources(source)) = &error else {
            panic!("image installation lost its typed GPU/I/O variants");
        };
        assert_eq!(source.to_string(), "coarse texture allocation failed");
        assert_eq!(
            error
                .source()
                .expect("runtime context retains the GPU error")
                .to_string(),
            "coarse texture allocation failed"
        );

        let validation = RuntimeError::Render(GpuOperationError::SurfaceValidation);
        assert_eq!(validation.category(), "render");
        assert_eq!(
            validation.to_string(),
            "failed to render the frame: surface texture acquisition failed validation"
        );
    }
}
