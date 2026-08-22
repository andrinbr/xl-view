//! Stateful GPU renderer: adapter/device and surface lifecycle, textures,
//! presentation resources, and submission. Pure output policy and textual
//! reports are delegated to [`super::output`] and [`super::diagnostics`].

use std::error::Error;
use std::fmt;
use std::io;
#[cfg(test)]
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use winit::dpi::PhysicalSize;
use winit::window::Window;

use super::diagnostics::{self, ImageDiagnostics, ImageWorkDiagnostics};
#[cfg(target_os = "linux")]
use super::display_target::HdrUnavailableReason;
use super::display_target::{DisplayTargetProvider, HdrTarget};
use super::hdr_metadata::{self, HdrMetadata, HdrSurface, SignalStatus};
use super::mip::{create_rgba16f_texture, mip_level_count, rgba16f_texture_budget_bytes};
#[cfg(test)]
use super::mip::{encode_premultiplied_rgba16f_row, generate_linear_mips_for_layers};
#[cfg(test)]
use super::output::SurfaceCandidate;
use super::output::{
    FALLBACK_HDR_SOURCE_PEAK_NITS, RenderingOptions, SurfaceOutputError, SurfaceSelection,
    is_hdr_color_space, select_required_surface, surface_configuration,
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
use super::{DisplayChangeNotifier, FailureReporter, NativeBackend, WorkReadyNotifier};
use crate::cli::{BackgroundMode, OutputMode};
use crate::units::bytes_to_mib;
use xl_view::color::{HDR_REFERENCE_WHITE_NITS, pq_oetf};
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
    pub(crate) fn backend_unavailable(
        message: &'static str,
        source: wgpu::RequestAdapterError,
    ) -> Self {
        Self::BackendUnavailable { message, source }
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DisplayRefresh {
    pub(crate) render_changed: bool,
    pub(crate) target_changed: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RenderOutcome {
    pub(crate) display_refresh: DisplayRefresh,
    pub(crate) presented: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SurfaceCapabilities {
    Unchanged,
    MayHaveChanged,
}

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
    source_peak_nits: f32,
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
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    display_target_provider: DisplayTargetProvider,
    instance: wgpu::Instance,
    output_mode: OutputMode,
    rendering_options: RenderingOptions,
    queue: wgpu::Queue,
    surface: HdrSurface<'static>,
    presentation: PresentationResources,
    image: ImageResources,
    ui_texture: wgpu::Texture,
    image_sampler: wgpu::Sampler,
    hdr_metadata_status: SignalStatus,
    gpu_memory_limit_bytes: u64,
    diagnostics_pattern: bool,
    notify_work_ready: WorkReadyNotifier,
    // Keep tile upload transactions atomic relative to presentation submission,
    // and prevent submission while surface configuration waits for idle.
    submission_lock: Arc<Mutex<()>>,
    selection: SurfaceSelection,
    #[cfg(target_os = "linux")]
    monitor_name: Option<String>,
    ui_centered: bool,
    window: Arc<dyn Window>,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Keeps surface/device/metadata initialization in one fallible transaction.
pub(super) fn initialize(
    instance: &wgpu::Instance,
    backend: NativeBackend,
    window: Arc<dyn Window>,
    output_mode: OutputMode,
    rendering_options: RenderingOptions,
    diagnostics_pattern: bool,
    report_failure: FailureReporter,
    notify_work_ready: WorkReadyNotifier,
    notify_display_changed: DisplayChangeNotifier,
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
    .map_err(|source| {
        GpuInitializationError::backend_unavailable(backend.unavailable_message(), source)
    })?;
    let display_target_provider = DisplayTargetProvider::new(notify_display_changed);
    #[cfg(target_os = "linux")]
    let monitor_name = current_monitor_name(window.as_ref());
    #[cfg(target_os = "linux")]
    let display_target = display_target_provider.target(monitor_name.as_deref());
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    let display_target = display_target_provider.target(&surface.display_hdr_info(&adapter));
    let capabilities = surface.get_capabilities(&adapter);
    let selection = select_surface_with_wayland_bootstrap(
        &capabilities.format_capabilities,
        output_mode,
        display_target,
        #[cfg(target_os = "linux")]
        monitor_name.as_deref(),
    )?;
    let candidate = selection.candidate();
    let device_descriptor = viewer_device_descriptor();
    let (device, queue, mut surface) = pollster::block_on(hdr_metadata::request_device(
        &adapter,
        &device_descriptor,
        surface,
    ))
    .map_err(|error| GpuInitializationError::DeviceCreation(Box::new(error)))?;
    install_failure_callbacks(&device, report_failure);
    let size = window.surface_size();
    let config = surface_configuration(candidate, size);
    let hdr_metadata_status = surface.configure(&config, output_hdr_metadata(selection));
    let adapter_info = adapter.get_info();
    tracing::info!(
        adapter = %adapter_info.name,
        device_type = ?adapter_info.device_type,
        backend = ?adapter_info.backend,
        initial_surface_format = ?candidate.format,
        initial_color_space = ?candidate.color_space,
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
        selection,
        rendering_options,
        None,
        unloaded_source_peak_nits(diagnostics_pattern),
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
        config,
        device,
        display_target_provider,
        instance: instance.clone(),
        output_mode,
        rendering_options,
        queue,
        surface,
        presentation,
        image: ImageResources::Fallback(fallback_image_bindings),
        ui_texture,
        image_sampler,
        hdr_metadata_status,
        gpu_memory_limit_bytes,
        diagnostics_pattern,
        notify_work_ready,
        submission_lock,
        selection,
        #[cfg(target_os = "linux")]
        monitor_name,
        ui_centered: false,
        window,
    })
}

fn select_surface_with_wayland_bootstrap(
    capabilities: &[wgpu::SurfaceFormatCapabilities],
    output_mode: OutputMode,
    display_target: HdrTarget,
    #[cfg(target_os = "linux")] monitor_name: Option<&str>,
) -> Result<SurfaceSelection, SurfaceOutputError> {
    #[cfg(target_os = "linux")]
    if output_mode == OutputMode::Pq
        && monitor_name.is_none()
        && display_target == Err(HdrUnavailableReason::OutputUnmatched)
    {
        // A new Wayland surface has no wl_output association until its first
        // commit. On multi-output systems that makes strict PQ selection
        // impossible during synchronous GPU initialization. Configure an sRGB
        // bootstrap surface so the compositor can identify the output; the
        // next display refresh then re-runs strict PQ selection before the
        // startup diagnostics report is emitted.
        tracing::debug!("waiting for Wayland to identify the window output before selecting PQ");
        return select_required_surface(capabilities, OutputMode::Sdr, display_target);
    }
    select_required_surface(capabilities, output_mode, display_target)
}

#[cfg(target_os = "linux")]
fn current_monitor_name(window: &dyn Window) -> Option<String> {
    window
        .current_monitor()
        .and_then(|monitor| monitor.name().map(Into::into))
}

fn viewer_device_descriptor() -> wgpu::DeviceDescriptor<'static> {
    let mut descriptor = wgpu::DeviceDescriptor::default();
    descriptor.required_features |= wgpu::Features::CLEAR_TEXTURE;
    descriptor.memory_hints = wgpu::MemoryHints::MemoryUsage;
    descriptor
}

fn output_hdr_metadata(selection: SurfaceSelection) -> Option<HdrMetadata> {
    is_hdr_color_space(selection.candidate().color_space)
        .then(|| HdrMetadata::display_mapped_bt2020(selection.output_peak_nits()))
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

const fn unloaded_source_peak_nits(diagnostics_pattern: bool) -> f32 {
    if diagnostics_pattern {
        HLG_PATTERN_SOURCE_PEAK_NITS
    } else {
        HDR_REFERENCE_WHITE_NITS
    }
}

impl GpuState {
    fn image_bindings(&self) -> &ImageBindings {
        self.image.bindings()
    }

    fn source_peak_nits(&self) -> f32 {
        self.image.loaded().map_or_else(
            || unloaded_source_peak_nits(self.diagnostics_pattern),
            |image| image.source_peak_nits,
        )
    }

    pub fn is_hdr_surface(&self) -> bool {
        is_hdr_color_space(self.selection.candidate().color_space)
    }

    pub fn image_gpu_budget_bytes(&self) -> u64 {
        self.image
            .loaded()
            .map_or(FALLBACK_IMAGE_GPU_BYTES, ImageState::gpu_budget_bytes)
    }

    pub fn ui_summary_rows(&self) -> Vec<(String, String)> {
        let candidate = self.selection.candidate();
        let mut rows = vec![(
            "Output surface".to_owned(),
            format!(
                "{} selected ({:?}/{:?})",
                if self.is_hdr_surface() { "HDR" } else { "SDR" },
                candidate.format,
                candidate.color_space,
            ),
        )];
        match self.selection.display_target() {
            Ok(peak_nits) => rows.push(("Display peak".to_owned(), format!("{peak_nits:.0} nits"))),
            Err(reason) => {
                rows.push(("Display peak".to_owned(), format!("Unavailable ({reason})")));
            }
        }
        #[cfg(target_os = "linux")]
        rows.push((
            "HDR metadata".to_owned(),
            self.hdr_metadata_summary().to_owned(),
        ));
        rows.push((
            "Exposure".to_owned(),
            format!("{:+.2} stops", self.rendering_options.exposure_stops),
        ));
        rows
    }

    #[cfg(target_os = "linux")]
    fn hdr_metadata_summary(&self) -> &'static str {
        match self.hdr_metadata_status {
            SignalStatus::Signaled => "Metadata signaled",
            SignalStatus::NotRequested => "Metadata not requested",
            SignalStatus::Unsupported => "Metadata unsupported",
            SignalStatus::SurfaceUnavailable => "Metadata unavailable",
        }
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
                source_peak_nits,
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
        let surface_selection = self.select_surface()?;
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
        if surface_selection.candidate().format == self.selection.candidate().format {
            self.rebuild_image_bind_group();
        }
        if surface_selection.candidate() == self.selection.candidate() {
            let output_peak_changed = surface_selection.display_target()
                != self.selection.display_target()
                && is_hdr_color_space(surface_selection.candidate().color_space);
            self.set_selection(surface_selection, "image changed");
            if output_peak_changed {
                self.refresh_hdr_metadata("display target changed while loading image");
            }
            self.update_params();
        } else {
            self.configure_surface(surface_selection, "display target changed");
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

    /// Returns whether this zoom snapped to the fit stop.
    pub fn zoom_at(&mut self, position: winit::dpi::PhysicalPosition<f64>, factor: f64) -> bool {
        let stopped_at_fit = self
            .image
            .loaded_mut()
            .is_some_and(|image| image.view_transform.zoom_at(position, factor));
        self.request_view_work();
        self.update_params();
        self.window.request_redraw();
        stopped_at_fit
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
        self.window.request_redraw();
    }

    pub fn reset_view_and_exposure(&mut self) {
        self.rendering_options.exposure_stops = 0.0;
        if let Some(image) = self.image.loaded_mut() {
            image.view_transform.set_fit();
        }
        self.request_view_work();
        self.update_params();
        self.window.request_redraw();
    }

    pub fn cycle_background(&mut self) {
        self.rendering_options.background = next_background(self.rendering_options.background);
        self.update_params();
        self.window.request_redraw();
    }

    pub fn refresh_display(
        &mut self,
        reason: &str,
        capabilities: SurfaceCapabilities,
    ) -> Result<DisplayRefresh, GpuOperationError> {
        self.refresh_surface_selection(reason, capabilities)
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn wayland_monitor_changed(&self) -> bool {
        self.monitor_name != current_monitor_name(self.window.as_ref())
    }

    pub fn process_background_work(&mut self) -> Result<(), io::Error> {
        let Some(image) = self.image.loaded_mut() else {
            return Ok(());
        };
        let mut visual_change = image.bindings.tile_cache.process_completions(&self.queue)?;
        let resampled_view_changed = image
            .bindings
            .viewport_resampler
            .process_completions(&self.device);
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
            .saturating_mul(image.store.tile_rows());
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

    pub fn render(&mut self) -> Result<RenderOutcome, GpuOperationError> {
        let display_refresh = if DisplayTargetProvider::QUERY_BEFORE_RENDER {
            self.refresh_surface_selection(
                "display target queried before render",
                SurfaceCapabilities::Unchanged,
            )?
        } else {
            DisplayRefresh::default()
        };
        let (surface_texture, suboptimal) = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(RenderOutcome {
                    display_refresh,
                    presented: false,
                });
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.reconfigure("surface became outdated")?;
                self.window.request_redraw();
                return Ok(RenderOutcome {
                    display_refresh,
                    presented: false,
                });
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.recreate_surface()?;
                self.window.request_redraw();
                return Ok(RenderOutcome {
                    display_refresh,
                    presented: false,
                });
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

        Ok(RenderOutcome {
            display_refresh,
            presented: true,
        })
    }

    #[cfg_attr(not(target_os = "linux"), allow(clippy::unnecessary_wraps))]
    pub fn startup_diagnostics_report(&self) -> Option<String> {
        // Windows and macOS expose their native HDR data through wgpu. Wayland
        // is queried by our separate color-management provider, so using wgpu's
        // empty DisplayHdrInfo there would hide the data that drove selection.
        #[cfg(target_os = "linux")]
        let raw_display_information = {
            // A Wayland window receives its output association only after the
            // first surface commit. Before then the report would describe a
            // potentially unrelated output.
            let monitor_name = self.monitor_name.as_deref()?;
            self.display_target_provider
                .raw_display_information(Some(monitor_name))
        };
        #[cfg(not(target_os = "linux"))]
        let raw_display_information =
            format!("{:#?}", self.surface.display_hdr_info(&self.adapter));
        Some(diagnostics::startup_report(
            &self.adapter,
            &self.surface,
            self.output_mode,
            self.selection,
            self.hdr_metadata_status,
            &raw_display_information,
            self.diagnostics_pattern,
        ))
    }

    pub fn image_diagnostics_report(&self) -> String {
        let loaded = self.image.loaded();
        diagnostics::image_report(ImageDiagnostics {
            background: self.rendering_options.background.as_str(),
            exposure_stops: self.rendering_options.exposure_stops,
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
            resampling_scratch_peak_bytes: bindings.viewport_resampler.scratch_peak_bytes(),
            gpu_image_budget_bytes: self.image_gpu_budget_bytes(),
            gpu_memory_limit_bytes: self.gpu_memory_limit_bytes,
        }
    }

    fn recreate_surface(&mut self) -> Result<(), GpuOperationError> {
        let surface = self.instance.create_surface(Arc::clone(&self.window))?;
        self.surface.replace(surface);
        self.reconfigure("surface was lost and recreated")
    }

    fn query_display_target(&self) -> HdrTarget {
        #[cfg(target_os = "linux")]
        {
            self.display_target_provider
                .target(self.monitor_name.as_deref())
        }
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        {
            let hdr_info = self.surface.display_hdr_info(&self.adapter);
            self.display_target_provider.target(&hdr_info)
        }
    }

    #[cfg_attr(not(target_os = "linux"), allow(clippy::unused_self))]
    fn refresh_monitor_name(&mut self) -> bool {
        #[cfg(target_os = "linux")]
        {
            let monitor_name = current_monitor_name(self.window.as_ref());
            let changed = monitor_name != self.monitor_name;
            self.monitor_name = monitor_name;
            changed
        }
        #[cfg(not(target_os = "linux"))]
        false
    }

    fn select_surface_for_target(
        &self,
        display_target: HdrTarget,
    ) -> Result<SurfaceSelection, SurfaceOutputError> {
        let capabilities = self.surface.get_capabilities(&self.adapter);
        select_surface_with_wayland_bootstrap(
            &capabilities.format_capabilities,
            self.output_mode,
            display_target,
            #[cfg(target_os = "linux")]
            self.monitor_name.as_deref(),
        )
    }

    fn select_surface(&self) -> Result<SurfaceSelection, SurfaceOutputError> {
        self.select_surface_for_target(self.query_display_target())
    }

    fn log_display_target_change(&self, display_target: HdrTarget, reason: &str) {
        let previous_display_target = self.selection.display_target();
        if display_target != previous_display_target {
            tracing::debug!(
                reason,
                ?previous_display_target,
                ?display_target,
                "display target changed"
            );
        }
    }

    fn set_selection(&mut self, selection: SurfaceSelection, reason: &str) {
        self.log_display_target_change(selection.display_target(), reason);
        self.selection = selection;
    }

    fn configure_surface(&mut self, selection: SurfaceSelection, reason: &str) {
        let candidate = selection.candidate();
        let previous_candidate = self.selection.candidate();
        let candidate_changed = candidate != previous_candidate;
        if candidate_changed {
            tracing::info!(
                reason,
                previous_format = ?previous_candidate.format,
                previous_color_space = ?previous_candidate.color_space,
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
        let format_changed = candidate.format != previous_candidate.format;
        self.set_selection(selection, reason);
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
        {
            let submission_lock = Arc::clone(&self.submission_lock);
            let _submission_guard = submission_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let metadata = output_hdr_metadata(self.selection);
            let status = self.surface.configure(&self.config, metadata);
            self.set_hdr_metadata_status(status, reason);
        }
        self.update_params();
    }

    fn reconfigure(&mut self, reason: &str) -> Result<(), GpuOperationError> {
        self.refresh_monitor_name();
        let selection = self.select_surface()?;
        self.configure_surface(selection, reason);
        Ok(())
    }

    fn refresh_surface_selection(
        &mut self,
        reason: &str,
        capabilities: SurfaceCapabilities,
    ) -> Result<DisplayRefresh, GpuOperationError> {
        let monitor_changed = self.refresh_monitor_name();
        let display_target = self.query_display_target();
        let target_changed = display_target != self.selection.display_target();
        if !target_changed && capabilities == SurfaceCapabilities::Unchanged && !monitor_changed {
            return Ok(DisplayRefresh::default());
        }
        let eligibility_changed = display_target.is_ok() != self.selection.display_target().is_ok();
        let selection = if capabilities == SurfaceCapabilities::MayHaveChanged
            || monitor_changed
            || (self.output_mode == OutputMode::Auto && eligibility_changed)
        {
            self.select_surface_for_target(display_target)
        } else {
            self.selection.with_display_target(display_target)
        };
        let selection = match selection {
            Ok(selection) => selection,
            Err(error) => {
                self.log_display_target_change(display_target, reason);
                return Err(error.into());
            }
        };
        Ok(DisplayRefresh {
            render_changed: self.apply_surface_selection(selection, reason),
            target_changed,
        })
    }

    fn apply_surface_selection(&mut self, selection: SurfaceSelection, reason: &str) -> bool {
        if selection.requires_surface_reconfiguration(self.selection) {
            self.configure_surface(selection, reason);
            true
        } else {
            // Compare the exact finite value written to the presentation uniform.
            let peak_changed = selection.output_peak_nits().to_bits()
                != self.selection.output_peak_nits().to_bits();
            self.set_selection(selection, reason);
            if peak_changed {
                if is_hdr_color_space(selection.candidate().color_space) {
                    self.refresh_hdr_metadata(reason);
                }
                self.update_params();
            }
            peak_changed
        }
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

    fn refresh_hdr_metadata(&mut self, reason: &str) {
        let status = self
            .surface
            .set_metadata(output_hdr_metadata(self.selection));
        self.set_hdr_metadata_status(status, reason);
    }

    fn update_params(&self) {
        let image_dimensions = self.image.loaded().map(|image| image.dimensions);
        let view_transform = self.image.loaded().map(|image| image.view_transform);
        let bindings = self.image_bindings();
        let mut params = shader_parameters(
            self.selection,
            self.rendering_options,
            image_dimensions,
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
    selection: SurfaceSelection,
    rendering_options: RenderingOptions,
    image_dimensions: Option<(u32, u32)>,
    source_peak_nits: f32,
    viewport_dimensions: (u32, u32),
    view_transform: Option<ViewTransform>,
    diagnostics_pattern: bool,
    ui_centered: bool,
) -> PresentationParams {
    let candidate = selection.candidate();
    let (image_width, image_height) = image_dimensions.unwrap_or((1, 1));
    let center = view_transform.map_or((0.5, 0.5), |view| {
        (view.center().x as f32, view.center().y as f32)
    });
    let view_scale = view_transform.map_or(1.0, |view| view.scale() as f32);
    let output_peak_nits = selection.output_peak_nits();
    let exposure = rendering_options.exposure_stops.exp2();
    let effective_source_peak_nits = HDR_REFERENCE_WHITE_NITS.max(source_peak_nits * exposure);
    PresentationParams {
        pq_output: is_hdr_color_space(candidate.color_space),
        content_mode: image_dimensions.map_or(u32::from(diagnostics_pattern), |_| 2),
        background_mode: background_shader_mode(rendering_options.background),
        hdr_reference_white_nits: HDR_REFERENCE_WHITE_NITS,
        source_peak_nits,
        output_peak_nits,
        exposure_stops: rendering_options.exposure_stops,
        effective_source_peak_pq: pq_oetf(effective_source_peak_nits / 10_000.0),
        output_peak_pq: pq_oetf(output_peak_nits / 10_000.0),
        viewport_width: viewport_dimensions.0,
        viewport_height: viewport_dimensions.1,
        image_width,
        image_height,
        view_center_x: center.0,
        view_center_y: center.1,
        view_scale,
        center_ui: ui_centered,
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
