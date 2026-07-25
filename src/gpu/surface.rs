//! Stateful GPU renderer: adapter/device and surface lifecycle, textures,
//! presentation resources, and submission. Pure output policy and textual
//! reports are delegated to [`super::output`] and [`super::diagnostics`].

use std::error::Error;
use std::fmt;
use std::io;
#[cfg(test)]
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use wgpu::SurfaceColorSpace;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use super::WorkReadyNotifier;
use super::diagnostics::{self, ImageDiagnostics, ImageWorkDiagnostics};
use super::hdr_metadata::{self, HdrMetadata, HdrSurface, SignalStatus};
use super::mip::{create_rgba16f_texture, mip_level_count, rgba16f_texture_budget_bytes};
#[cfg(test)]
use super::mip::{encode_premultiplied_rgba16f_row, generate_linear_mips_for_layers};
#[cfg(test)]
use super::output::{
    EXTENDED_LINEAR_FALLBACK_PEAK_NITS, HLG_ENCODING_PEAK_NITS, PQ_ENCODING_PEAK_NITS,
};
use super::output::{
    FALLBACK_HDR_SOURCE_PEAK_NITS, RenderingOptions, SurfaceCandidate, SurfaceOutputError,
    finite_non_negative, has_hdr_encoding_candidate, hdr_mapping_summary, is_hdr_color_space,
    quantization_bits, resolved_output_peak_nits, resolved_ui_white_nits,
    select_required_surface_candidate, shader_mode, surface_configuration, surface_hdr_metadata,
};
#[cfg(test)]
use super::presentation::{PARAM_BUFFER_USAGE, presentation_shader_source};
use super::presentation::{PresentationParams, PresentationResources};
use super::resampling::{ViewportResampler, required_resampling_gpu_bytes};
#[cfg(test)]
use super::tiles::{
    MISSING_TILE_SLOT, TileJob, TileWorkerState, create_tile_array_texture,
    create_tile_mapping_buffer, interactive_working_set_tiles, maximum_capacity, prioritized_tiles,
    replace_pending_jobs, same_tile_set, upload_tile_layer,
};
use super::tiles::{TileCache, tile_texture_extent, working_set_capacity};
use super::ui::{OverlayImage, OverlaySection, render_empty_state, render_text_overlay};
use super::upload::upload_staging_buffer_bytes;
use super::view::ViewTransform;
use crate::cli::{BackgroundMode, OutputMode};
use crate::units::bytes_to_mib;
use xl_view::color::HDR_REFERENCE_WHITE_NITS;
use xl_view::decode::{DecodedImage, DecodedTileStore, SourceDynamicRange, TILE_GUTTER, TILE_SIZE};

const HLG_PATTERN_SOURCE_PEAK_NITS: f32 = 1_000.0;
const FALLBACK_IMAGE_GPU_BYTES: u64 = 8;
const FALLBACK_UPLOAD_STAGING_BUFFER_BYTES: usize = 256;
#[derive(Debug)]
pub enum GpuFailure {
    DeviceLost {
        reason: wgpu::DeviceLostReason,
        message: String,
    },
    OutOfMemory,
    Internal(String),
    Validation(String),
}

impl fmt::Display for GpuFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceLost { reason, message } if message.is_empty() => {
                write!(formatter, "the GPU device was lost ({reason:?}); restart xl-view")
            }
            Self::DeviceLost { reason, message } => {
                write!(formatter, "the GPU device was lost ({reason:?}): {message}; restart xl-view")
            }
            Self::OutOfMemory => formatter.write_str(
                "the GPU ran out of memory; close other GPU-intensive applications or open a smaller image",
            ),
            Self::Internal(message) => write!(formatter, "an internal GPU error occurred: {message}"),
            Self::Validation(message) => {
                write!(formatter, "the GPU rejected a rendering operation: {message}")
            }
        }
    }
}

impl Error for GpuFailure {}

impl GpuFailure {
    pub const fn category(&self) -> &'static str {
        match self {
            Self::DeviceLost { .. } => "gpu_device_lost",
            Self::OutOfMemory => "gpu_out_of_memory",
            Self::Internal(_) => "gpu_internal",
            Self::Validation(_) => "gpu_validation",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum GpuInitializationError {
    #[error("{message} ({source})")]
    BackendUnavailable {
        message: &'static str,
        #[source]
        source: wgpu::RequestAdapterError,
    },

    #[error(transparent)]
    OutputUnavailable(#[from] SurfaceOutputError),

    #[error("failed to create the GPU presentation surface: {0}")]
    SurfaceCreation(#[source] wgpu::CreateSurfaceError),

    #[error("failed to create the GPU device: {0}")]
    DeviceCreation(#[source] Box<dyn Error>),
}

impl GpuInitializationError {
    pub(crate) fn backend_unavailable(source: wgpu::RequestAdapterError) -> Self {
        Self::BackendUnavailable {
            message: super::backend_unavailable_message(),
            source,
        }
    }

    pub(crate) const fn category(&self) -> &'static str {
        match self {
            Self::BackendUnavailable { .. } => "gpu_backend_unavailable",
            Self::OutputUnavailable(_) => "gpu_output_unavailable",
            Self::SurfaceCreation(_) | Self::DeviceCreation(_) => "gpu_initialization",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum GpuOperationError {
    #[error(transparent)]
    ImageResources(#[from] io::Error),

    #[error(transparent)]
    OutputUnavailable(#[from] SurfaceOutputError),

    #[error(transparent)]
    SurfaceCreation(#[from] wgpu::CreateSurfaceError),

    #[error("surface texture acquisition failed validation")]
    SurfaceValidation,
}

type FailureReporter = Arc<dyn Fn(GpuFailure) + Send + Sync>;

/// Shader-visible resources that exist for both the empty and loaded states.
///
/// The fallback state owns one set because every presentation bind group must
/// remain complete before an image is installed; replacing that state drops
/// the fallback allocations.
struct ImageBindings {
    coarse_texture: wgpu::Texture,
    tile_cache: TileCache,
    viewport_resampler: ViewportResampler,
}

/// All mutable GPU state derived from one installed decoded image.
///
/// Replacing this value commits the store, dimensions, view, bindings, HDR
/// source properties, and accounting together, so none can refer to different
/// image generations.
struct ImageState {
    bindings: ImageBindings,
    store: Arc<DecodedTileStore>,
    dimensions: (u32, u32),
    view_transform: ViewTransform,
    source_dynamic_range: SourceDynamicRange,
    source_peak_nits: f32,
    source_min_nits: f32,
    base_gpu_budget_bytes: u64,
    coarse_gpu_bytes: u64,
    resampling_memory_shortfall: Option<(u64, u64)>,
    cpu_storage_bytes: usize,
    upload_staging_buffer_bytes: usize,
    coarse_mip_levels: u32,
}

impl ImageState {
    fn gpu_budget_bytes(&self) -> u64 {
        self.base_gpu_budget_bytes
            .saturating_add(self.bindings.viewport_resampler.gpu_bytes())
    }
}

enum ImageResources {
    Fallback(ImageBindings),
    Loaded(ImageState),
}

impl ImageResources {
    const fn bindings(&self) -> &ImageBindings {
        match self {
            Self::Fallback(bindings) => bindings,
            Self::Loaded(image) => &image.bindings,
        }
    }

    const fn loaded(&self) -> Option<&ImageState> {
        match self {
            Self::Fallback(_) => None,
            Self::Loaded(image) => Some(image),
        }
    }

    const fn loaded_mut(&mut self) -> Option<&mut ImageState> {
        match self {
            Self::Fallback(_) => None,
            Self::Loaded(image) => Some(image),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResamplingReservation {
    NotRequired,
    Reserved(u64),
    Insufficient { required: u64, available: u64 },
}

#[derive(Clone, Copy)]
struct SurfaceSelection {
    candidate: SurfaceCandidate,
    hdr_encoding_available: bool,
}

/// A fully validated image transition whose remaining commit steps are infallible.
///
/// Preparation only borrows the renderer, so a failure drops staged resources
/// without disturbing the currently installed image or surface selection.
struct PreparedImageInstall {
    image: ImageState,
    resampling_reservation: ResamplingReservation,
    surface_selection: SurfaceSelection,
}

impl ResamplingReservation {
    const fn bytes(self) -> u64 {
        match self {
            Self::Reserved(bytes) => bytes,
            Self::NotRequired | Self::Insufficient { .. } => 0,
        }
    }
}

pub struct GpuState {
    adapter: wgpu::Adapter,
    candidate: SurfaceCandidate,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    instance: wgpu::Instance,
    last_hdr_info: wgpu::DisplayHdrInfo,
    output_mode: OutputMode,
    rendering_options: RenderingOptions,
    queue: wgpu::Queue,
    surface: HdrSurface<'static>,
    presentation: PresentationResources,
    image: ImageResources,
    ui_texture: wgpu::Texture,
    image_sampler: wgpu::Sampler,
    hdr_metadata_status: SignalStatus,
    hdr_encoding_available: bool,
    gpu_memory_limit_bytes: u64,
    diagnostics_pattern: bool,
    notify_work_ready: WorkReadyNotifier,
    // Keep tile upload transactions atomic relative to presentation submission,
    // and prevent submission while surface configuration waits for idle.
    submission_lock: Arc<Mutex<()>>,
    ui_centered: bool,
    window: Arc<dyn Window>,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Keeps surface/device/metadata initialization in one fallible transaction.
pub fn initialize(
    instance: &wgpu::Instance,
    window: Arc<dyn Window>,
    output_mode: OutputMode,
    rendering_options: RenderingOptions,
    diagnostics_pattern: bool,
    report_failure: FailureReporter,
    notify_work_ready: WorkReadyNotifier,
    gpu_memory_limit_bytes: u64,
) -> Result<GpuState, GpuInitializationError> {
    let surface = instance
        .create_surface(Arc::clone(&window))
        .map_err(GpuInitializationError::SurfaceCreation)?;
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        ..Default::default()
    }))
    .map_err(GpuInitializationError::backend_unavailable)?;
    let capabilities = surface.get_capabilities(&adapter);
    let source_dynamic_range = if diagnostics_pattern {
        SourceDynamicRange::Hlg
    } else {
        SourceDynamicRange::Sdr
    };
    let candidate = select_required_surface_candidate(
        &capabilities.format_capabilities,
        output_mode,
        source_dynamic_range,
    )?;
    let hdr_encoding_available = has_hdr_encoding_candidate(&capabilities.format_capabilities);

    let device_descriptor = viewer_device_descriptor();
    let (device, queue, hdr_metadata_signaler) =
        pollster::block_on(hdr_metadata::request_device(&adapter, &device_descriptor))
            .map_err(|error| GpuInitializationError::DeviceCreation(Box::new(error)))?;
    let metadata_supported = hdr_metadata_signaler.is_supported();
    let mut surface = hdr_metadata_signaler.bind_surface(surface);
    install_failure_callbacks(&device, report_failure);
    let size = window.surface_size();
    let config = surface_configuration(candidate, size);
    let last_hdr_info = surface.display_hdr_info(&adapter);
    let initial_metadata = surface_hdr_metadata(
        candidate,
        rendering_options,
        source_dynamic_range,
        HLG_PATTERN_SOURCE_PEAK_NITS,
        0.0,
    );
    let hdr_metadata_status = surface.configure(&config, initial_metadata);
    let adapter_info = adapter.get_info();
    tracing::info!(
        adapter = %adapter_info.name,
        device_type = ?adapter_info.device_type,
        backend = ?adapter_info.backend,
        surface_format = ?candidate.format,
        color_space = ?candidate.color_space,
        metadata_supported,
        metadata_status = %hdr_metadata_status,
        "GPU surface initialized"
    );

    let coarse_texture = create_coarse_texture(&device, &queue, &[0.0; 4], 1, 1)
        .expect("a one-pixel coarse-preview fallback texture must fit");
    let tile_cache = TileCache::fallback(&device);
    let viewport_resampler = ViewportResampler::fallback(&device, 0);
    let fallback_image_bindings = ImageBindings {
        coarse_texture,
        tile_cache,
        viewport_resampler,
    };
    let ui_texture = create_ui_texture(&device, &queue, &[0.0; 4], 1, 1)
        .expect("a one-pixel transparent UI intermediate must fit");
    let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("image linear sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    let submission_lock = Arc::new(Mutex::new(()));
    let params = shader_parameters(
        candidate,
        rendering_options,
        &last_hdr_info,
        None,
        source_dynamic_range,
        HLG_PATTERN_SOURCE_PEAK_NITS,
        (config.width, config.height),
        None,
        diagnostics_pattern,
        false,
    );
    let presentation = PresentationResources::new(
        &device,
        candidate.format,
        params,
        &fallback_image_bindings.coarse_texture,
        &fallback_image_bindings.tile_cache,
        fallback_image_bindings.viewport_resampler.texture(),
        &ui_texture,
        &image_sampler,
    );

    Ok(GpuState {
        adapter,
        candidate,
        config,
        device,
        instance: instance.clone(),
        last_hdr_info,
        output_mode,
        rendering_options,
        queue,
        surface,
        presentation,
        image: ImageResources::Fallback(fallback_image_bindings),
        ui_texture,
        image_sampler,
        hdr_metadata_status,
        hdr_encoding_available,
        gpu_memory_limit_bytes,
        diagnostics_pattern,
        notify_work_ready,
        submission_lock,
        ui_centered: false,
        window,
    })
}

fn viewer_device_descriptor() -> wgpu::DeviceDescriptor<'static> {
    let mut descriptor = wgpu::DeviceDescriptor::default();
    descriptor.required_features |= wgpu::Features::CLEAR_TEXTURE;
    descriptor.memory_hints = wgpu::MemoryHints::MemoryUsage;
    descriptor
}

fn install_failure_callbacks(device: &wgpu::Device, report_failure: FailureReporter) {
    let report_device_loss = Arc::clone(&report_failure);
    device.set_device_lost_callback(move |reason, message| {
        report_device_loss(GpuFailure::DeviceLost { reason, message });
    });

    device.on_uncaptured_error(Arc::new(move |error| {
        let failure = match error {
            wgpu::Error::OutOfMemory { .. } => GpuFailure::OutOfMemory,
            wgpu::Error::Internal { description, .. } => GpuFailure::Internal(description),
            wgpu::Error::Validation { description, .. } => GpuFailure::Validation(description),
        };
        report_failure(failure);
    }));
}

fn create_coarse_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
) -> Result<wgpu::Texture, io::Error> {
    create_rgba16f_texture(
        device,
        queue,
        pixels,
        width,
        height,
        mip_level_count(width, height),
        "coarse premultiplied linear BT.2020 preview",
    )
}

fn create_ui_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
) -> Result<wgpu::Texture, io::Error> {
    create_rgba16f_texture(
        device,
        queue,
        pixels,
        width,
        height,
        1,
        "premultiplied linear-sRGB UI",
    )
}

impl GpuState {
    fn image_bindings(&self) -> &ImageBindings {
        self.image.bindings()
    }

    fn source_dynamic_range(&self) -> SourceDynamicRange {
        self.image.loaded().map_or(
            if self.diagnostics_pattern {
                SourceDynamicRange::Hlg
            } else {
                SourceDynamicRange::Sdr
            },
            |image| image.source_dynamic_range,
        )
    }

    fn source_peak_nits(&self) -> f32 {
        self.image
            .loaded()
            .map_or(HLG_PATTERN_SOURCE_PEAK_NITS, |image| image.source_peak_nits)
    }

    fn source_min_nits(&self) -> f32 {
        self.image
            .loaded()
            .map_or(0.0, |image| image.source_min_nits)
    }

    pub fn is_hdr_surface(&self) -> bool {
        is_hdr_color_space(self.candidate.color_space)
    }

    pub fn hdr_encoding_unavailable(&self) -> bool {
        !self.hdr_encoding_available
    }

    pub fn image_gpu_budget_bytes(&self) -> u64 {
        self.image
            .loaded()
            .map_or(FALLBACK_IMAGE_GPU_BYTES, ImageState::gpu_budget_bytes)
    }

    pub fn hdr_metadata_summary(&self) -> &'static str {
        match self.hdr_metadata_status {
            #[cfg(not(target_vendor = "apple"))]
            SignalStatus::Signaled => "Metadata signaled",
            SignalStatus::NotRequested => "Metadata not requested",
            #[cfg(not(target_vendor = "apple"))]
            SignalStatus::Unsupported => "Metadata unsupported",
            #[cfg(not(target_vendor = "apple"))]
            SignalStatus::SurfaceUnavailable => "Metadata unavailable",
            #[cfg(target_vendor = "apple")]
            SignalStatus::BackendManaged => "Metadata managed by Metal",
        }
    }

    pub fn ui_summary_rows(&self) -> Vec<(String, String)> {
        let display_capability = [
            self.last_hdr_info
                .luminance
                .and_then(|luminance| luminance.max_nits)
                .map(|nits| format!("peak {nits:.0} nits")),
            self.last_hdr_info
                .tone_map_headroom()
                .map(|value| format!("headroom {value:.2}x")),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(", ");
        let mut rows = vec![(
            "Output surface".to_owned(),
            format!(
                "{} selected ({:?}/{:?})",
                if self.is_hdr_surface() { "HDR" } else { "SDR" },
                self.candidate.format,
                self.candidate.color_space,
            ),
        )];
        if !display_capability.is_empty() {
            rows.push(("Display capability".to_owned(), display_capability));
        }
        rows.extend([
            (
                "HDR mapping".to_owned(),
                hdr_mapping_summary(self.is_hdr_surface(), self.source_dynamic_range()).to_owned(),
            ),
            (
                "HDR metadata".to_owned(),
                self.hdr_metadata_summary().to_owned(),
            ),
            (
                "Exposure".to_owned(),
                format!("{:+.2} stops", self.rendering_options.exposure_stops),
            ),
        ]);
        rows
    }

    pub fn set_metadata_overlay(&mut self, sections: &[OverlaySection]) -> Result<(), io::Error> {
        let overlay = render_text_overlay(sections, self.window.scale_factor());
        self.set_ui_overlay(&overlay, false)
    }

    pub fn set_empty_state(&mut self) -> Result<(), io::Error> {
        let overlay = render_empty_state(self.window.scale_factor());
        self.set_ui_overlay(&overlay, true)
    }

    fn set_ui_overlay(&mut self, overlay: &OverlayImage, centered: bool) -> Result<(), io::Error> {
        self.ui_texture = create_ui_texture(
            &self.device,
            &self.queue,
            &overlay.pixels,
            overlay.width,
            overlay.height,
        )?;
        self.rebuild_image_bind_group();
        self.ui_centered = centered;
        self.update_params();
        self.window.request_redraw();
        Ok(())
    }

    fn create_image_state(
        &self,
        image: &DecodedImage,
        view_transform: ViewTransform,
    ) -> Result<(ImageState, ResamplingReservation), io::Error> {
        let store = &image.store;
        let (coarse_width, coarse_height) = store.coarse_dimensions();
        let coarse_gpu_bytes = rgba16f_texture_budget_bytes(
            coarse_width,
            coarse_height,
            mip_level_count(coarse_width, coarse_height),
            1,
        );
        if coarse_gpu_bytes > self.gpu_memory_limit_bytes {
            return Err(io::Error::other(format!(
                "the {} MiB GPU memory budget cannot hold the {:.2} MiB coarse image preview",
                self.gpu_memory_limit_bytes / (1024 * 1024),
                bytes_to_mib(coarse_gpu_bytes),
            )));
        }
        let coarse_texture = create_coarse_texture(
            &self.device,
            &self.queue,
            store.coarse_pixels(),
            coarse_width,
            coarse_height,
        )?;
        let resampling_reservation = resampling_reservation(
            (self.config.width, self.config.height),
            Some(view_transform),
            self.device.limits().max_texture_dimension_2d,
            self.gpu_memory_limit_bytes.saturating_sub(coarse_gpu_bytes),
        );
        let resampling_gpu_bytes = resampling_reservation.bytes();
        let tile_gpu_bytes = self
            .gpu_memory_limit_bytes
            .saturating_sub(coarse_gpu_bytes)
            .saturating_sub(resampling_gpu_bytes);
        let tile_cache = TileCache::active(
            &self.device,
            &self.queue,
            Arc::clone(store),
            tile_gpu_bytes,
            (self.config.width, self.config.height),
            Arc::clone(&self.notify_work_ready),
            Arc::clone(&self.submission_lock),
        )?;
        let viewport_resampler = ViewportResampler::active(
            &self.device,
            &self.queue,
            Arc::clone(store),
            resampling_gpu_bytes,
            Arc::clone(&self.notify_work_ready),
            Arc::clone(&self.submission_lock),
        );
        let source_peak_nits = if image.source_dynamic_range == SourceDynamicRange::Sdr {
            HDR_REFERENCE_WHITE_NITS
        } else {
            let metadata_peak = image.metadata.tone_mapping.intensity_target_nits;
            if metadata_peak.is_finite() && metadata_peak > 0.0 {
                metadata_peak
            } else {
                FALLBACK_HDR_SOURCE_PEAK_NITS
            }
        };
        let base_gpu_budget_bytes = coarse_gpu_bytes.saturating_add(tile_cache.gpu_bytes());
        let tile_extent = tile_texture_extent();
        let upload_staging_buffer_bytes = upload_staging_buffer_bytes(coarse_width, coarse_height)
            .max(upload_staging_buffer_bytes(tile_extent, tile_extent));
        Ok((
            ImageState {
                bindings: ImageBindings {
                    coarse_texture,
                    tile_cache,
                    viewport_resampler,
                },
                store: Arc::clone(store),
                dimensions: (image.width, image.height),
                view_transform,
                source_dynamic_range: image.source_dynamic_range,
                source_peak_nits,
                source_min_nits: finite_non_negative(image.metadata.tone_mapping.min_nits),
                base_gpu_budget_bytes,
                coarse_gpu_bytes,
                resampling_memory_shortfall: None,
                cpu_storage_bytes: image.memory_cost_bytes,
                upload_staging_buffer_bytes,
                coarse_mip_levels: mip_level_count(coarse_width, coarse_height),
            },
            resampling_reservation,
        ))
    }

    fn prepare_image_install(
        &self,
        image: &DecodedImage,
    ) -> Result<PreparedImageInstall, GpuOperationError> {
        let width = image.width;
        let height = image.height;
        let view_transform = ViewTransform::fit(
            PhysicalSize::new(width, height),
            PhysicalSize::new(self.config.width, self.config.height),
        );
        let (image_state, resampling_reservation) =
            self.create_image_state(image, view_transform)?;
        let surface_selection = self.select_surface(image.source_dynamic_range)?;
        Ok(PreparedImageInstall {
            image: image_state,
            resampling_reservation,
            surface_selection,
        })
    }

    fn commit_image_install(&mut self, prepared: PreparedImageInstall) {
        let PreparedImageInstall {
            image,
            resampling_reservation,
            surface_selection,
        } = prepared;
        self.image = ImageResources::Loaded(image);
        self.note_resampling_reservation(
            (self.config.width, self.config.height),
            resampling_reservation,
        );
        if let Some(image) = self.image.loaded_mut() {
            image
                .bindings
                .tile_cache
                .request_view(&self.queue, image.view_transform);
        }
        let _ = self.request_resampling();
        if surface_selection.candidate.format == self.candidate.format {
            self.rebuild_image_bind_group();
        }
        if surface_selection.candidate == self.candidate {
            self.hdr_encoding_available = surface_selection.hdr_encoding_available;
            self.update_params();
            self.refresh_hdr_metadata("image changed");
        } else {
            self.configure_surface(surface_selection, "source dynamic range changed");
        }
    }

    pub fn set_image(&mut self, image: &Arc<DecodedImage>) -> Result<(), GpuOperationError> {
        let prepared = self.prepare_image_install(image)?;
        self.commit_image_install(prepared);
        self.window.request_redraw();
        Ok(())
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) -> Result<(), GpuOperationError> {
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        self.config.width = size.width;
        self.config.height = size.height;
        if let Some(image) = self.image.loaded_mut() {
            image.view_transform.set_viewport(size);
        }
        self.request_view_work();
        self.reconfigure("window resized")?;
        Ok(())
    }

    pub fn scale_factor_changed(
        &mut self,
        size: PhysicalSize<u32>,
        scale_factor: f64,
    ) -> Result<(), GpuOperationError> {
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        self.config.width = size.width;
        self.config.height = size.height;
        if let Some(image) = self.image.loaded_mut() {
            image.view_transform.set_viewport(size);
            image.view_transform.set_scale_factor(scale_factor);
        }
        self.request_view_work();
        self.reconfigure("window scale factor changed")?;
        Ok(())
    }

    pub fn fit_view(&mut self) {
        if let Some(image) = self.image.loaded_mut() {
            image.view_transform.set_fit();
        }
        self.request_view_work();
        self.update_params();
        self.window.request_redraw();
    }

    pub fn one_to_one(&mut self, scale_factor: f64) {
        if let Some(image) = self.image.loaded_mut() {
            image.view_transform.set_one_to_one(scale_factor);
        }
        self.request_view_work();
        self.update_params();
        self.window.request_redraw();
    }

    pub fn zoom_at(&mut self, position: winit::dpi::PhysicalPosition<f64>, factor: f64) {
        if let Some(image) = self.image.loaded_mut() {
            image.view_transform.zoom_at(position, factor);
        }
        self.request_view_work();
        self.update_params();
        self.window.request_redraw();
    }

    pub fn pan_by(&mut self, delta_x: f64, delta_y: f64) {
        if let Some(image) = self.image.loaded_mut() {
            image.view_transform.pan_by(delta_x, delta_y);
        }
        self.request_view_work();
        self.update_params();
        self.window.request_redraw();
    }

    pub fn adjust_exposure(&mut self, delta_stops: f32) {
        self.rendering_options.exposure_stops =
            (self.rendering_options.exposure_stops + delta_stops).clamp(-10.0, 10.0);
        self.update_params();
        self.refresh_hdr_metadata("exposure changed");
        self.window.request_redraw();
    }

    pub fn reset_view_and_exposure(&mut self) {
        self.rendering_options.exposure_stops = 0.0;
        if let Some(image) = self.image.loaded_mut() {
            image.view_transform.set_fit();
        }
        self.request_view_work();
        self.update_params();
        self.refresh_hdr_metadata("exposure reset");
        self.window.request_redraw();
    }

    pub fn cycle_background(&mut self) {
        self.rendering_options.background = next_background(self.rendering_options.background);
        self.update_params();
        self.window.request_redraw();
    }

    pub fn refresh_display(&mut self, reason: &str) -> Result<(), GpuOperationError> {
        self.reconfigure(reason)?;
        self.window.request_redraw();
        Ok(())
    }

    pub fn process_background_work(&mut self) -> Result<(), io::Error> {
        let Some(image) = self.image.loaded_mut() else {
            return Ok(());
        };
        let mut visual_change = image.bindings.tile_cache.process_completions(&self.queue)?;
        let resampled_view_changed = image.bindings.viewport_resampler.process_completions();
        if resampled_view_changed {
            self.rebuild_image_bind_group();
            debug_assert!(self.image_gpu_budget_bytes() <= self.gpu_memory_limit_bytes);
            self.update_params();
            visual_change = true;
        }
        if visual_change {
            self.window.request_redraw();
        }
        Ok(())
    }

    fn request_view_work(&mut self) {
        if self.image.loaded().is_none() {
            return;
        }
        let resampling_gpu_bytes = self.current_resampling_reservation_bytes();
        if let Some(image) = self.image.loaded_mut() {
            image
                .bindings
                .viewport_resampler
                .set_memory_limit_bytes(resampling_gpu_bytes);
        }
        let resampling_binding_changed = self.request_resampling();
        let tile_binding_changed = self.rebudget_tile_cache(resampling_gpu_bytes);
        if let Some(image) = self.image.loaded_mut() {
            image
                .bindings
                .tile_cache
                .request_view(&self.queue, image.view_transform);
        }
        if resampling_binding_changed || tile_binding_changed {
            self.rebuild_image_bind_group();
        }
    }

    fn request_resampling(&mut self) -> bool {
        let Some(image) = self.image.loaded_mut() else {
            return false;
        };
        image.bindings.viewport_resampler.request_view(
            &self.device,
            (self.config.width, self.config.height),
            Some(image.view_transform),
            self.device.limits().max_texture_dimension_2d,
        )
    }

    fn current_resampling_reservation_bytes(&mut self) -> u64 {
        let Some(image) = self.image.loaded() else {
            return 0;
        };
        let view_transform = image.view_transform;
        let available_gpu_bytes = self
            .gpu_memory_limit_bytes
            .saturating_sub(image.coarse_gpu_bytes);
        self.resampling_reservation_bytes_with_warning(
            (self.config.width, self.config.height),
            Some(view_transform),
            self.device.limits().max_texture_dimension_2d,
            available_gpu_bytes,
        )
    }

    fn resampling_reservation_bytes_with_warning(
        &mut self,
        viewport_dimensions: (u32, u32),
        view_transform: Option<ViewTransform>,
        maximum_texture_dimension: u32,
        available_gpu_bytes: u64,
    ) -> u64 {
        let reservation = resampling_reservation(
            viewport_dimensions,
            view_transform,
            maximum_texture_dimension,
            available_gpu_bytes,
        );
        self.note_resampling_reservation(viewport_dimensions, reservation);
        reservation.bytes()
    }

    fn note_resampling_reservation(
        &mut self,
        viewport_dimensions: (u32, u32),
        reservation: ResamplingReservation,
    ) {
        match reservation {
            ResamplingReservation::Insufficient {
                required,
                available,
            } => {
                let Some(image) = self.image.loaded_mut() else {
                    return;
                };
                if image.resampling_memory_shortfall.is_none() {
                    tracing::warn!(
                        viewport_width = viewport_dimensions.0,
                        viewport_height = viewport_dimensions.1,
                        required_mib = bytes_to_mib(required),
                        available_mib = bytes_to_mib(available),
                        gpu_memory_limit_mib = bytes_to_mib(self.gpu_memory_limit_bytes),
                        "GPU memory budget cannot hold the resampled viewport; using image tiles or the coarse preview"
                    );
                }
                image.resampling_memory_shortfall = Some((required, available));
            }
            ResamplingReservation::NotRequired | ResamplingReservation::Reserved(_) => {
                if let Some(image) = self.image.loaded_mut() {
                    image.resampling_memory_shortfall = None;
                }
            }
        }
    }

    fn rebudget_tile_cache(&mut self, resampling_gpu_bytes: u64) -> bool {
        let Some(image) = self.image.loaded_mut() else {
            return false;
        };
        let tile_gpu_bytes = self
            .gpu_memory_limit_bytes
            .saturating_sub(image.coarse_gpu_bytes)
            .saturating_sub(resampling_gpu_bytes);
        let logical_tiles = image
            .store
            .tile_columns()
            .checked_mul(image.store.tile_rows())
            .unwrap_or(u32::MAX);
        let desired_capacity = working_set_capacity(
            logical_tiles,
            (self.config.width, self.config.height),
            tile_gpu_bytes,
            self.device.limits().max_texture_array_layers,
        );
        if desired_capacity == image.bindings.tile_cache.capacity() {
            return false;
        }
        let tile_cache = match TileCache::active(
            &self.device,
            &self.queue,
            Arc::clone(&image.store),
            tile_gpu_bytes,
            (self.config.width, self.config.height),
            Arc::clone(&self.notify_work_ready),
            Arc::clone(&self.submission_lock),
        ) {
            Ok(cache) => cache,
            Err(error) => {
                tracing::warn!(%error, "cannot resize the GPU tile cache; using the coarse preview");
                TileCache::fallback(&self.device)
            }
        };
        image.bindings.tile_cache = tile_cache;
        image.base_gpu_budget_bytes = image
            .coarse_gpu_bytes
            .saturating_add(image.bindings.tile_cache.gpu_bytes());
        debug_assert!(image.gpu_budget_bytes() <= self.gpu_memory_limit_bytes);
        true
    }

    fn rebuild_image_bind_group(&mut self) {
        let bindings = self.image.bindings();
        self.presentation.rebuild_bind_group(
            &self.device,
            &bindings.coarse_texture,
            &bindings.tile_cache,
            bindings.viewport_resampler.texture(),
            &self.ui_texture,
            &self.image_sampler,
        );
    }

    pub fn render(&mut self) -> Result<bool, GpuOperationError> {
        let (surface_texture, suboptimal) = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.reconfigure("surface became outdated")?;
                self.window.request_redraw();
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.recreate_surface()?;
                self.window.request_redraw();
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(GpuOperationError::SurfaceValidation);
            }
        };
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("presentation encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("presentation render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.presentation.pipeline);
            pass.set_bind_group(0, &self.presentation.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        let command_buffer = encoder.finish();
        {
            // Keep presentation submission from splitting a worker's tile
            // upload and mip-generation submission into separate transactions.
            let _submission_guard = self
                .submission_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.queue.submit([command_buffer]);
            self.window.pre_present_notify();
            self.queue.present(surface_texture);
        }

        if suboptimal {
            self.reconfigure("surface became suboptimal")?;
            self.window.request_redraw();
        }

        Ok(true)
    }

    pub fn startup_diagnostics_report(&self) -> String {
        diagnostics::startup_report(
            &self.adapter,
            &self.surface,
            self.output_mode,
            self.source_dynamic_range(),
            self.candidate,
            self.hdr_metadata_status,
            self.diagnostics_pattern,
        )
    }

    pub fn image_diagnostics_report(&self) -> String {
        let loaded = self.image.loaded();
        diagnostics::image_report(ImageDiagnostics {
            background: self.rendering_options.background.as_str(),
            exposure_stops: self.rendering_options.exposure_stops,
            hdr_metadata_status: self.hdr_metadata_status,
            cpu_storage_bytes: loaded.map_or(0, |image| image.cpu_storage_bytes),
            upload_staging_buffer_bytes: loaded
                .map_or(FALLBACK_UPLOAD_STAGING_BUFFER_BYTES, |image| {
                    image.upload_staging_buffer_bytes
                }),
            allocator_report: self.device.generate_allocator_report(),
            coarse_mip_levels: loaded.map_or(1, |image| image.coarse_mip_levels),
            work: self.image_work_diagnostics(),
        })
    }

    pub fn image_finished_diagnostics_report(&self) -> String {
        diagnostics::image_finished_report(&self.image_work_diagnostics())
    }

    fn image_work_diagnostics(&self) -> ImageWorkDiagnostics {
        let bindings = self.image_bindings();
        ImageWorkDiagnostics {
            tile_cache: bindings.tile_cache.status(),
            tile_hits: bindings.tile_cache.hits,
            tile_misses: bindings.tile_cache.misses,
            viewport_resampler: bindings.viewport_resampler.status(),
            gpu_image_budget_bytes: self.image_gpu_budget_bytes(),
            gpu_memory_limit_bytes: self.gpu_memory_limit_bytes,
        }
    }

    fn recreate_surface(&mut self) -> Result<(), GpuOperationError> {
        let surface = self.instance.create_surface(Arc::clone(&self.window))?;
        self.surface.replace(surface);
        self.reconfigure("surface was lost and recreated")
    }

    fn select_surface(
        &self,
        source_dynamic_range: SourceDynamicRange,
    ) -> Result<SurfaceSelection, SurfaceOutputError> {
        let capabilities = self.surface.get_capabilities(&self.adapter);
        let candidate = select_required_surface_candidate(
            &capabilities.format_capabilities,
            self.output_mode,
            source_dynamic_range,
        )?;
        Ok(SurfaceSelection {
            candidate,
            hdr_encoding_available: has_hdr_encoding_candidate(&capabilities.format_capabilities),
        })
    }

    fn configure_surface(&mut self, selection: SurfaceSelection, reason: &str) {
        let SurfaceSelection {
            candidate,
            hdr_encoding_available,
        } = selection;
        let candidate_changed = candidate != self.candidate;
        if candidate_changed {
            tracing::info!(
                reason,
                previous_format = ?self.candidate.format,
                previous_color_space = ?self.candidate.color_space,
                format = ?candidate.format,
                color_space = ?candidate.color_space,
                "surface output changed"
            );
        }

        let current_size = self.window.surface_size();
        if current_size.width > 0 && current_size.height > 0 {
            self.config.width = current_size.width;
            self.config.height = current_size.height;
        }
        self.config = surface_configuration(
            candidate,
            PhysicalSize::new(self.config.width, self.config.height),
        );
        let hdr_info = self.surface.display_hdr_info(&self.adapter);
        let hdr_info_changed = hdr_info != self.last_hdr_info;
        if hdr_info_changed {
            tracing::info!(reason, hdr_info = ?hdr_info, "display HDR information changed");
        }
        let format_changed = candidate.format != self.candidate.format;
        self.candidate = candidate;
        self.hdr_encoding_available = hdr_encoding_available;
        self.last_hdr_info = hdr_info;
        if format_changed {
            let bindings = self.image.bindings();
            self.presentation.rebuild_pipeline(
                &self.device,
                candidate.format,
                &bindings.coarse_texture,
                &bindings.tile_cache,
                bindings.viewport_resampler.texture(),
                &self.ui_texture,
                &self.image_sampler,
            );
        }
        self.update_params();
        let metadata = self.current_hdr_metadata();
        let submission_lock = Arc::clone(&self.submission_lock);
        let _submission_guard = submission_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let status = self.surface.configure(&self.config, metadata);
        self.set_hdr_metadata_status(status, reason);
    }

    fn reconfigure(&mut self, reason: &str) -> Result<(), GpuOperationError> {
        let selection = self.select_surface(self.source_dynamic_range())?;
        self.configure_surface(selection, reason);
        Ok(())
    }

    fn current_hdr_metadata(&self) -> Option<HdrMetadata> {
        surface_hdr_metadata(
            self.candidate,
            self.rendering_options,
            self.source_dynamic_range(),
            self.source_peak_nits(),
            self.source_min_nits(),
        )
    }

    fn refresh_hdr_metadata(&mut self, reason: &str) {
        let metadata = self.current_hdr_metadata();
        let status = self.surface.set_metadata(metadata);
        self.set_hdr_metadata_status(status, reason);
    }

    fn set_hdr_metadata_status(&mut self, status: SignalStatus, reason: &str) {
        if status != self.hdr_metadata_status {
            tracing::info!(
                reason,
                previous_status = %self.hdr_metadata_status,
                %status,
                "HDR metadata state changed"
            );
        }
        self.hdr_metadata_status = status;
    }

    fn update_params(&self) {
        let image_dimensions = self.image.loaded().map(|image| image.dimensions);
        let view_transform = self.image.loaded().map(|image| image.view_transform);
        let bindings = self.image_bindings();
        let mut params = shader_parameters(
            self.candidate,
            self.rendering_options,
            &self.last_hdr_info,
            image_dimensions,
            self.source_dynamic_range(),
            self.source_peak_nits(),
            (self.config.width, self.config.height),
            view_transform,
            self.diagnostics_pattern,
            self.ui_centered,
        );
        params.tiled_image = bindings.tile_cache.should_sample(view_transform);
        params.tile_columns = bindings.tile_cache.tile_columns();
        params.resampled_viewport = bindings.viewport_resampler.is_active();
        let params = params.to_ne_bytes();
        self.queue
            .write_buffer(&self.presentation.params_buffer, 0, params.as_flattened());
    }
}

fn resampling_reservation(
    viewport_dimensions: (u32, u32),
    view_transform: Option<ViewTransform>,
    maximum_texture_dimension: u32,
    available_gpu_bytes: u64,
) -> ResamplingReservation {
    let Some(required) = view_transform.and_then(|view| {
        required_resampling_gpu_bytes(viewport_dimensions, view.scale(), maximum_texture_dimension)
    }) else {
        return ResamplingReservation::NotRequired;
    };
    if required <= available_gpu_bytes {
        ResamplingReservation::Reserved(required)
    } else {
        ResamplingReservation::Insufficient {
            required,
            available: available_gpu_bytes,
        }
    }
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)] // GPU uniforms intentionally keep explicit f32 coordinates and rendering inputs.
fn shader_parameters(
    candidate: SurfaceCandidate,
    rendering_options: RenderingOptions,
    hdr_info: &wgpu::DisplayHdrInfo,
    image_dimensions: Option<(u32, u32)>,
    source_dynamic_range: SourceDynamicRange,
    source_peak_nits: f32,
    viewport_dimensions: (u32, u32),
    view_transform: Option<ViewTransform>,
    diagnostics_pattern: bool,
    ui_centered: bool,
) -> PresentationParams {
    let (image_width, image_height) = image_dimensions.unwrap_or((1, 1));
    let center = view_transform.map_or((0.5, 0.5), |view| {
        (view.center().x as f32, view.center().y as f32)
    });
    let view_scale = view_transform.map_or(1.0, |view| view.scale() as f32);
    let output_peak_nits = resolved_output_peak_nits(candidate.color_space, hdr_info);
    PresentationParams {
        mode: shader_mode(candidate.color_space),
        encode_srgb: candidate.color_space == SurfaceColorSpace::Srgb
            && !candidate.format.is_srgb(),
        dither_bits: u32::from(quantization_bits(candidate.format)),
        content_mode: image_dimensions.map_or(u32::from(diagnostics_pattern), |_| 2),
        hdr_reference_white_nits: HDR_REFERENCE_WHITE_NITS,
        source_peak_nits,
        output_peak_nits,
        exposure_stops: rendering_options.exposure_stops,
        viewport_width: viewport_dimensions.0,
        viewport_height: viewport_dimensions.1,
        image_width,
        image_height,
        ui_white_nits: resolved_ui_white_nits(hdr_info),
        view_center_x: center.0,
        view_center_y: center.1,
        view_scale,
        background_mode: background_shader_mode(rendering_options.background),
        center_ui: ui_centered,
        source_dynamic_range: source_dynamic_range.shader_code(),
        tiled_image: false,
        tile_size: TILE_SIZE,
        tile_columns: 0,
        tile_gutter: TILE_GUTTER,
        resampled_viewport: false,
    }
}

fn background_shader_mode(background: BackgroundMode) -> u32 {
    match background {
        BackgroundMode::Black => 0,
        BackgroundMode::Checkerboard => 1,
        BackgroundMode::White => 2,
        BackgroundMode::MiddleGray => 3,
    }
}

fn next_background(background: BackgroundMode) -> BackgroundMode {
    match background {
        BackgroundMode::Black => BackgroundMode::MiddleGray,
        BackgroundMode::MiddleGray => BackgroundMode::White,
        BackgroundMode::White => BackgroundMode::Checkerboard,
        BackgroundMode::Checkerboard => BackgroundMode::Black,
    }
}

#[cfg(test)]
mod tests;
