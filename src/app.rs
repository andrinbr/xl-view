//! Winit application coordinator for window, input, file-transfer, event, and
//! GPU lifecycle. Image selection and decode/cache/prefetch state live in
//! [`session::ImageSession`].

#[cfg(target_os = "linux")]
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
#[cfg(test)]
use winit::cursor::CursorIcon;
use winit::data_transfer::{DataTransferId, TypeHint, TypedData};
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{
    ButtonSource, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::run_on_demand::EventLoopExtRunOnDemand;
use winit::event_loop::{
    ActiveEventLoop, AsyncRequestSerial, ControlFlow, DndAction, EventLoop, EventLoopProxy,
};
use winit::keyboard::ModifiersState;
#[cfg(test)]
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::monitor::Fullscreen;
#[cfg(target_os = "linux")]
use winit::platform::{
    startup_notify::{self, EventLoopExtStartupNotify},
    wayland::{EventLoopBuilderExtWayland, WindowAttributesWayland},
};
use winit::window::{Window, WindowAttributes, WindowId};

#[cfg(target_os = "linux")]
use crate::APPLICATION_ID;
use crate::APPLICATION_NAME;
use crate::app_icon;
use crate::cli::{BackgroundMode, Cli, OutputMode};
use crate::diagnostics;
use crate::error::{AppError, RuntimeError};
use crate::gpu;
use xl_view::color::HDR_REFERENCE_WHITE_NITS;
use xl_view::decode::{DecodeCompletion, DecodeCoordinator, DecodeLimits, ImageKey};
#[cfg(test)]
use xl_view::metadata::{ExifMetadata, XmpMetadata};

mod decoded_cache;
mod files;
mod input;
mod overlay;
mod session;

use self::decoded_cache::automatic_decoded_cache_bytes;
use self::files::{
    FilePickerResult, FolderDirection, NeighborPaths, adjacent_image_path, first_image_path,
    neighboring_image_paths, select_image_file,
};
#[cfg(test)]
use self::files::{choose_adjacent_path, file_name};
use self::input::{
    FullscreenState, KeyboardAction, image_cursor, is_double_click, keyboard_action,
};
#[cfg(test)]
use self::overlay::{DecodeTiming, format_exif_datetime, source_encoding_details, xmp_rating_row};
use self::overlay::{
    attribution_metadata_rows, capture_metadata_rows, decode_summary, dimensions_summary,
    source_range_summary,
};
use self::session::{
    DecodeEffect, ImageSession, NavigationEffect, PendingImageInstall, SelectionEffect,
};

const FULLSCREEN_CURSOR_HIDE_DELAY: Duration = Duration::from_secs(2);

struct Application {
    background: BackgroundMode,
    cursor_hide_deadline: Option<Instant>,
    cursor_position: Option<PhysicalPosition<f64>>,
    cursor_visibility: CursorVisibility,
    diagnostics: bool,
    fatal_error: Option<RuntimeError>,
    event_receiver: Receiver<UserEvent>,
    event_sender: UserEventSender,
    file_picker_state: FilePickerState,
    fullscreen: FullscreenState,
    gpu: Option<gpu::GpuState>,
    instance: wgpu::Instance,
    is_dragging: bool,
    last_drag_position: Option<PhysicalPosition<f64>>,
    last_primary_click: Option<Instant>,
    metadata_visible: bool,
    modifiers: ModifiersState,
    output_mode: OutputMode,
    pending_drop: Option<PendingDrop>,
    session: ImageSession,
    gpu_memory_limit_bytes: u64,
    window: Option<Arc<dyn Window>>,
}

#[derive(Clone)]
struct UserEventSender {
    sender: Sender<UserEvent>,
    proxy: EventLoopProxy,
}

impl UserEventSender {
    fn send(&self, event: UserEvent) {
        if self.sender.send(event).is_ok() {
            self.proxy.wake_up();
        }
    }
}

#[derive(Debug)]
struct PendingDrop {
    id: DataTransferId,
    fetch_serial: AsyncRequestSerial,
    dropped: bool,
    data: PendingDropData,
}

#[derive(Debug)]
enum PendingDropData {
    Awaiting,
    Path(PathBuf),
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FilePickerState {
    Closed,
    Open,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CursorVisibility {
    Hidden,
    Visible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OverlayEffect {
    None,
    Refresh,
}

impl PendingDrop {
    fn take_ready_data(&mut self) -> Option<PendingDropData> {
        if !self.dropped || matches!(self.data, PendingDropData::Awaiting) {
            return None;
        }
        Some(std::mem::replace(&mut self.data, PendingDropData::Awaiting))
    }
}

impl Application {
    fn select_path(&mut self, path: PathBuf, direction: Option<FolderDirection>) {
        match self.session.select_path(path, direction) {
            SelectionEffect::Install(install) => self.install_image(install),
            SelectionEffect::Pending | SelectionEffect::StatusChanged => {
                self.update_window_title();
            }
        }
    }

    fn install_image(&mut self, install: PendingImageInstall) {
        let previous_image_report = self.current_image_finished_report("replaced");
        if let Some(gpu) = self.gpu.as_mut() {
            if let Err(error) = gpu.set_image(install.image()) {
                tracing::warn!(
                    path = %install.path().display(),
                    purpose = "demand",
                    selection_generation = install.generation(),
                    direction = ?install.direction(),
                    %error,
                    "decoded image cannot be installed on the GPU"
                );
                self.session.reject_install(&install, &error);
                self.update_window_title();
                return;
            }
            if let Some(report) = previous_image_report {
                diagnostics::print_report("image-finished", &report);
            }
        }

        let retain_decoded_image = self.gpu.is_none();
        self.session.commit_install(install, retain_decoded_image);
        self.update_window_title();
        self.refresh_metadata_overlay();
        self.update_cursor();
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn start_neighbor_lookup(&mut self) {
        let Some(request) = self.session.begin_neighbor_lookup() else {
            return;
        };
        let generation = request.generation;
        let anchor_key = request.anchor_key.clone();
        let anchor_path = request.anchor_path.clone();
        let event_sender = self.event_sender.clone();
        let spawn = std::thread::Builder::new()
            .name("xl-view-neighbor-prefetch-lookup".to_owned())
            .spawn(move || {
                let result = neighboring_image_paths(&anchor_path);
                event_sender.send(UserEvent::NeighborPathsComplete {
                    generation,
                    anchor_key,
                    result,
                });
            });
        if let Err(error) = spawn {
            self.session.neighbor_lookup_start_failed(&request, &error);
        }
    }

    fn navigate_folder(&mut self, direction: FolderDirection) {
        let Some(request) = self.session.begin_navigation(direction) else {
            self.update_window_title();
            return;
        };
        self.update_window_title();
        let generation = request.generation;
        let lookup_direction = request.direction;
        let lookup_path = request.source_path.clone();
        let event_sender = self.event_sender.clone();
        let spawn = std::thread::Builder::new()
            .name("xl-view-folder-navigation".to_owned())
            .spawn(move || {
                let result = adjacent_image_path(&lookup_path, lookup_direction);
                event_sender.send(UserEvent::FolderNavigationComplete {
                    generation,
                    direction: lookup_direction,
                    source_path: lookup_path,
                    result,
                });
            });
        if let Err(error) = spawn {
            self.session.navigation_start_failed(&request, &error);
            self.update_window_title();
        }
    }

    fn open_file_picker(&mut self) {
        if self.file_picker_state == FilePickerState::Open {
            return;
        }
        let Some(window) = self.window.as_ref().map(Arc::clone) else {
            return;
        };
        let current_folder = self
            .session
            .current_path()
            .and_then(Path::parent)
            .filter(|path| !path.as_os_str().is_empty())
            .map(Path::to_path_buf);
        let event_sender = self.event_sender.clone();
        let spawn = std::thread::Builder::new()
            .name("xl-view-file-picker".to_owned())
            .spawn(move || {
                let result = select_image_file(window.as_ref(), current_folder);
                event_sender.send(UserEvent::FilePickerComplete(result));
            });
        match spawn {
            Ok(_) => self.file_picker_state = FilePickerState::Open,
            Err(error) => {
                tracing::warn!(%error, "cannot start the file picker");
                self.session.set_status_message(format!(
                    "Cannot open the file picker: {error}; drop an image or open one from the command line"
                ));
                self.update_window_title();
            }
        }
    }

    fn handle_file_picker_result(&mut self, result: FilePickerResult) {
        self.file_picker_state = FilePickerState::Closed;
        match result {
            FilePickerResult::Selected(path) => self.select_path(path, None),
            FilePickerResult::Cancelled => {}
            FilePickerResult::Failed(error) => {
                tracing::warn!(%error, "file picker failed");
                self.session.set_status_message(format!(
                    "Cannot open the selected file: {error}; choose an image"
                ));
                self.update_window_title();
            }
        }
    }

    fn update_cursor(&self) {
        if let Some(window) = self.window.as_ref() {
            window
                .set_cursor(image_cursor(self.session.has_loaded_image(), self.is_dragging).into());
        }
    }

    fn set_cursor_visibility(&mut self, visible: bool) {
        let visibility = if visible {
            CursorVisibility::Visible
        } else {
            CursorVisibility::Hidden
        };
        if self.cursor_visibility == visibility {
            return;
        }
        self.cursor_visibility = visibility;
        if let Some(window) = self.window.as_ref() {
            window.set_cursor_visible(visible);
        }
    }

    fn note_pointer_activity(&mut self, now: Instant) {
        self.set_cursor_visibility(true);
        self.cursor_hide_deadline =
            fullscreen_cursor_hide_deadline(self.fullscreen, self.is_dragging, now);
    }

    fn show_cursor_without_timeout(&mut self) {
        self.cursor_hide_deadline = None;
        self.set_cursor_visibility(true);
    }

    fn hide_fullscreen_cursor_if_due(&mut self, now: Instant) {
        if self
            .cursor_hide_deadline
            .is_some_and(|deadline| now >= deadline)
        {
            self.cursor_hide_deadline = None;
            self.set_cursor_visibility(false);
        }
    }

    fn stop_dragging(&mut self) {
        self.is_dragging = false;
        self.last_drag_position = None;
        self.update_cursor();
    }

    fn toggle_fullscreen(&mut self) {
        self.set_fullscreen(self.fullscreen.toggled());
    }

    fn set_fullscreen(&mut self, fullscreen: FullscreenState) {
        self.fullscreen = fullscreen;
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.fit_view();
        }
        if let Some(window) = self.window.as_ref() {
            window.set_fullscreen(
                (fullscreen == FullscreenState::Fullscreen).then_some(Fullscreen::Borderless(None)),
            );
        }
        self.note_pointer_activity(Instant::now());
    }

    fn update_window_title(&self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let title = if let Some(message) = self.session.status_message() {
            format!("{APPLICATION_NAME} - {message}")
        } else if let Some(image) = self.session.loaded_image() {
            if image.hdr_source && self.gpu.as_ref().is_some_and(|gpu| !gpu.is_hdr_surface()) {
                format!(
                    "{APPLICATION_NAME} - {} - {}x{} - HDR image shown in standard range",
                    image.file_name, image.dimensions.0, image.dimensions.1
                )
            } else {
                format!(
                    "{APPLICATION_NAME} - {} - {}x{}",
                    image.file_name, image.dimensions.0, image.dimensions.1
                )
            }
        } else {
            format!("{APPLICATION_NAME} - No image open")
        };
        window.set_title(&title);
    }

    fn metadata_sections(&self) -> Vec<gpu::OverlaySection> {
        let Some(image) = self.session.loaded_image() else {
            return Vec::new();
        };
        let mut sections = vec![
            gpu::OverlaySection {
                title: "IMAGE",
                rows: vec![
                    ("File".to_owned(), image.file_name.clone()),
                    (
                        "Dimensions".to_owned(),
                        dimensions_summary(image.dimensions.0, image.dimensions.1),
                    ),
                    (
                        "Decode".to_owned(),
                        decode_summary(image.decode_timing, image.memory_bytes),
                    ),
                ],
            },
            gpu::OverlaySection {
                title: "SOURCE",
                rows: vec![
                    (
                        "Range".to_owned(),
                        source_range_summary(image.hdr_source, &image.source_transfer),
                    ),
                    ("Color space".to_owned(), image.source_color_space.clone()),
                    (
                        "Intensity target".to_owned(),
                        format!("{:.0} nits", image.source_intensity_nits),
                    ),
                ],
            },
        ];
        let display_rows = if let Some(gpu) = self.gpu.as_ref() {
            gpu.ui_summary_rows()
        } else {
            vec![
                ("Output surface".to_owned(), "pending".to_owned()),
                ("HDR metadata".to_owned(), "Metadata pending".to_owned()),
            ]
        };
        sections.push(gpu::OverlaySection {
            title: "DISPLAY & VIEWER",
            rows: display_rows,
        });

        let capture_rows = capture_metadata_rows(image.exif.as_ref());
        if !capture_rows.is_empty() {
            sections.push(gpu::OverlaySection {
                title: "CAPTURE",
                rows: capture_rows,
            });
        }
        let attribution_rows = attribution_metadata_rows(image.exif.as_ref(), image.xmp.as_ref());
        if !attribution_rows.is_empty() {
            sections.push(gpu::OverlaySection {
                title: "ATTRIBUTION",
                rows: attribution_rows,
            });
        }
        sections
    }

    fn refresh_metadata_overlay(&mut self) {
        if !self.session.has_loaded_image() && !self.diagnostics {
            if let Some(gpu) = self.gpu.as_mut()
                && let Err(error) = gpu.set_empty_state()
            {
                tracing::warn!(%error, "cannot render the welcome screen");
                self.session
                    .set_status_message(format!("Cannot show the welcome screen: {error}"));
                self.update_window_title();
            }
            return;
        }
        let sections = if self.metadata_visible {
            self.metadata_sections()
        } else {
            Vec::new()
        };
        if let Some(gpu) = self.gpu.as_mut()
            && let Err(error) = gpu.set_metadata_overlay(&sections)
        {
            tracing::warn!(%error, "cannot render the metadata overlay");
            self.session
                .set_status_message(format!("Cannot update metadata overlay: {error}"));
            self.update_window_title();
        }
    }

    fn begin_file_drop(&mut self, event_loop: &dyn ActiveEventLoop, id: DataTransferId) {
        self.pending_drop = None;

        let transfer = match event_loop.data_transfer(id) {
            Ok(transfer) => transfer,
            Err(error) => {
                tracing::warn!(%error, "cannot inspect dragged data");
                return;
            }
        };
        if !transfer.has_type(&TypeHint::UriList) {
            let _ = event_loop.set_valid_dnd_actions(id, &[]);
            return;
        }
        if let Err(error) = event_loop.set_valid_dnd_actions(id, &[DndAction::Copy]) {
            tracing::warn!(%error, "cannot accept dragged file paths");
            return;
        }

        match event_loop.fetch_data_transfer(id, &TypeHint::UriList) {
            Ok(fetch_serial) => {
                self.pending_drop = Some(PendingDrop {
                    id,
                    fetch_serial,
                    dropped: false,
                    data: PendingDropData::Awaiting,
                });
            }
            Err(error) => {
                let _ = event_loop.set_valid_dnd_actions(id, &[]);
                tracing::warn!(%error, "cannot read dragged file paths");
            }
        }
    }

    fn finish_pending_drop_if_ready(&mut self) {
        let data = self
            .pending_drop
            .as_mut()
            .and_then(PendingDrop::take_ready_data);
        let Some(data) = data else {
            return;
        };
        self.pending_drop = None;
        match data {
            PendingDropData::Path(path) => self.select_path(path, None),
            PendingDropData::Unsupported => {
                self.session
                    .set_status_message("The dropped data does not contain a supported image");
                self.update_window_title();
            }
            PendingDropData::Awaiting => unreachable!("a ready drop has received its data"),
        }
    }

    fn handle_decode_completion(&mut self, completion: DecodeCompletion) {
        match self.session.handle_decode_completion(completion) {
            DecodeEffect::Install(install) => self.install_image(install),
            DecodeEffect::StatusChanged => self.update_window_title(),
            DecodeEffect::None => {}
        }
    }

    fn handle_user_event(&mut self, event_loop: &dyn ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::DecodeComplete(completion) => self.handle_decode_completion(*completion),
            UserEvent::FilePickerComplete(result) => self.handle_file_picker_result(result),
            UserEvent::FolderNavigationComplete {
                generation,
                direction,
                source_path,
                result,
            } => match self.session.handle_navigation_result(
                generation,
                direction,
                &source_path,
                result,
            ) {
                NavigationEffect::Select { path, direction } => {
                    self.select_path(path, Some(direction));
                }
                NavigationEffect::StatusChanged => self.update_window_title(),
                NavigationEffect::None => {}
            },
            UserEvent::NeighborPathsComplete {
                generation,
                anchor_key,
                result,
            } => self
                .session
                .handle_neighbor_paths(generation, &anchor_key, result),
            UserEvent::GpuFailure(failure) => {
                self.fatal_error = Some(failure.into());
                event_loop.exit();
            }
            UserEvent::GpuWorkReady => {
                if let Some(gpu) = self.gpu.as_mut()
                    && let Err(error) = gpu.process_background_work()
                {
                    self.fatal_error = Some(RuntimeError::GpuBackgroundWork(error));
                    event_loop.exit();
                }
            }
        }
    }

    fn zoom_anchor(&self) -> Option<PhysicalPosition<f64>> {
        self.cursor_position.or_else(|| {
            self.window.as_ref().map(|window| {
                let size = window.surface_size();
                PhysicalPosition::new(f64::from(size.width) / 2.0, f64::from(size.height) / 2.0)
            })
        })
    }

    #[allow(clippy::too_many_lines)] // One exhaustive match intentionally owns every keyboard action.
    fn handle_keyboard_action(&mut self, event_loop: &dyn ActiveEventLoop, action: KeyboardAction) {
        let overlay_effect = match action {
            KeyboardAction::Quit => {
                event_loop.exit();
                return;
            }
            KeyboardAction::Open => {
                self.open_file_picker();
                OverlayEffect::None
            }
            KeyboardAction::ToggleMetadata => {
                self.metadata_visible = !self.metadata_visible;
                OverlayEffect::Refresh
            }
            KeyboardAction::ToggleFullscreen => {
                self.toggle_fullscreen();
                OverlayEffect::None
            }
            KeyboardAction::LeaveFullscreen => {
                self.set_fullscreen(FullscreenState::Windowed);
                OverlayEffect::None
            }
            KeyboardAction::PreviousImage => {
                self.navigate_folder(FolderDirection::Previous);
                OverlayEffect::None
            }
            KeyboardAction::NextImage => {
                self.navigate_folder(FolderDirection::Next);
                OverlayEffect::None
            }
            KeyboardAction::Fit => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.fit_view();
                }
                OverlayEffect::None
            }
            KeyboardAction::OneToOne => {
                let scale_factor = self.window.as_ref().map(|window| window.scale_factor());
                if let (Some(gpu), Some(scale_factor)) = (self.gpu.as_mut(), scale_factor) {
                    gpu.one_to_one(scale_factor);
                }
                OverlayEffect::None
            }
            KeyboardAction::ZoomIn => {
                let zoom_anchor = self.zoom_anchor();
                if let (Some(gpu), Some(position)) = (self.gpu.as_mut(), zoom_anchor) {
                    gpu.zoom_at(position, 1.25);
                }
                OverlayEffect::None
            }
            KeyboardAction::ZoomOut => {
                let zoom_anchor = self.zoom_anchor();
                if let (Some(gpu), Some(position)) = (self.gpu.as_mut(), zoom_anchor) {
                    gpu.zoom_at(position, 0.8);
                }
                OverlayEffect::None
            }
            KeyboardAction::ExposureDown => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.adjust_exposure(-0.25);
                }
                OverlayEffect::Refresh
            }
            KeyboardAction::ExposureUp => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.adjust_exposure(0.25);
                }
                OverlayEffect::Refresh
            }
            KeyboardAction::ResetViewAndExposure => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.reset_view_and_exposure();
                }
                OverlayEffect::Refresh
            }
            KeyboardAction::CycleBackground => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.cycle_background();
                }
                OverlayEffect::None
            }
            KeyboardAction::PanLeft => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.pan_by(64.0, 0.0);
                }
                OverlayEffect::None
            }
            KeyboardAction::PanRight => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.pan_by(-64.0, 0.0);
                }
                OverlayEffect::None
            }
            KeyboardAction::PanUp => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.pan_by(0.0, 64.0);
                }
                OverlayEffect::None
            }
            KeyboardAction::PanDown => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.pan_by(0.0, -64.0);
                }
                OverlayEffect::None
            }
        };
        self.update_window_title();
        if overlay_effect == OverlayEffect::Refresh {
            self.refresh_metadata_overlay();
        }
    }

    fn handle_surface_resized(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        size: PhysicalSize<u32>,
    ) {
        if let Some(gpu) = self.gpu.as_mut()
            && let Err(error) = gpu.resize(size)
        {
            self.fatal_error = Some(RuntimeError::SurfaceResize(error));
            event_loop.exit();
            return;
        }
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn handle_scale_factor_changed(&mut self, event_loop: &dyn ActiveEventLoop, scale_factor: f64) {
        if let (Some(gpu), Some(window)) = (self.gpu.as_mut(), self.window.as_ref()) {
            if let Err(error) = gpu.scale_factor_changed(window.surface_size(), scale_factor) {
                self.fatal_error = Some(RuntimeError::SurfaceScaleFactor(error));
                event_loop.exit();
            } else {
                window.request_redraw();
                self.refresh_metadata_overlay();
            }
        }
    }

    fn handle_window_moved(&mut self, event_loop: &dyn ActiveEventLoop) {
        if let Some(gpu) = self.gpu.as_mut()
            && let Err(error) = gpu.refresh_display("window moved to another output")
        {
            self.fatal_error = Some(RuntimeError::DisplayMove(error));
            event_loop.exit();
        }
        self.refresh_metadata_overlay();
    }

    fn handle_focus_changed(&mut self, event_loop: &dyn ActiveEventLoop, focused: bool) {
        if !focused {
            self.handle_pointer_departure();
            return;
        }
        if let Some(gpu) = self.gpu.as_mut()
            && let Err(error) = gpu.refresh_display("window regained focus")
        {
            self.fatal_error = Some(RuntimeError::DisplayFocus(error));
            event_loop.exit();
        }
        self.update_window_title();
        self.refresh_metadata_overlay();
        self.note_pointer_activity(Instant::now());
    }

    fn handle_file_dropped(&mut self, id: DataTransferId) {
        if let Some(pending) = self.pending_drop.as_mut()
            && pending.id == id
        {
            pending.dropped = true;
            self.finish_pending_drop_if_ready();
        }
    }

    fn handle_file_drag_left(&mut self, id: DataTransferId) {
        if self
            .pending_drop
            .as_ref()
            .is_some_and(|pending| pending.id == id)
        {
            self.pending_drop = None;
        }
    }

    fn handle_data_transfer_received(
        &mut self,
        id: DataTransferId,
        serial: AsyncRequestSerial,
        value: &dyn TypedData,
    ) {
        if let Some(pending) = self.pending_drop.as_mut()
            && pending.id == id
            && pending.fetch_serial == serial
        {
            pending.data = match value.try_as_file_paths() {
                Ok(paths) => first_image_path(&paths)
                    .map_or(PendingDropData::Unsupported, PendingDropData::Path),
                Err(error) => {
                    tracing::warn!(%error, "cannot decode dropped file paths");
                    PendingDropData::Unsupported
                }
            };
            self.finish_pending_drop_if_ready();
        }
    }

    fn handle_pointer_departure(&mut self) {
        self.stop_dragging();
        self.show_cursor_without_timeout();
    }

    fn handle_pointer_entered(&mut self, position: PhysicalPosition<f64>) {
        self.cursor_position = Some(position);
        self.note_pointer_activity(Instant::now());
    }

    fn handle_pointer_moved(&mut self, position: PhysicalPosition<f64>) {
        self.cursor_position = Some(position);
        if self.is_dragging {
            if let (Some(previous), Some(gpu)) = (self.last_drag_position, self.gpu.as_mut()) {
                gpu.pan_by(position.x - previous.x, position.y - previous.y);
            }
            self.last_drag_position = Some(position);
        }
        self.note_pointer_activity(Instant::now());
    }

    fn handle_pointer_button(&mut self, state: ElementState, button: &ButtonSource) {
        if matches!(button, ButtonSource::Mouse(MouseButton::Left)) {
            self.handle_primary_pointer_button(state);
        }
        self.note_pointer_activity(Instant::now());
    }

    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        self.note_pointer_activity(Instant::now());
        match delta {
            // Winit generally reports discrete mouse wheels in lines and touchpad scrolling in
            // pixels. Keeping line scrolling as zoom and pixel scrolling as pan avoids treating a
            // two-finger touchpad pan as zoom. A smooth-scroll mouse that only reports pixels will
            // pan as well because Winit does not expose the input source here.
            // see https://github.com/rust-windowing/winit/issues/4315.
            MouseScrollDelta::LineDelta(_, vertical) => {
                let amount = f64::from(vertical) * 0.18;
                if let (Some(position), Some(gpu)) = (self.cursor_position, self.gpu.as_mut()) {
                    gpu.zoom_at(position, amount.exp());
                }
            }
            MouseScrollDelta::PixelDelta(delta) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.pan_by(delta.x, delta.y);
                }
            }
            _ => {}
        }
    }

    fn handle_pinch_gesture(&mut self, delta: f64) {
        self.note_pointer_activity(Instant::now());
        let anchor = self.zoom_anchor();
        if let (Some(gpu), Some(anchor)) = (self.gpu.as_mut(), anchor) {
            // Winit reports an additive magnification delta; zoom_at expects a factor.
            gpu.zoom_at(anchor, 1.0 + delta);
        }
    }

    fn handle_pan_gesture(&mut self, delta: PhysicalPosition<f32>) {
        self.note_pointer_activity(Instant::now());
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.pan_by(f64::from(delta.x), f64::from(delta.y));
        }
    }

    fn handle_primary_pointer_button(&mut self, state: ElementState) {
        if state != ElementState::Pressed {
            self.stop_dragging();
            return;
        }
        let now = Instant::now();
        if self
            .last_primary_click
            .is_some_and(|previous| is_double_click(previous, now))
        {
            self.last_primary_click = None;
            self.stop_dragging();
            self.toggle_fullscreen();
            return;
        }
        self.last_primary_click = Some(now);
        self.is_dragging = self.session.has_loaded_image();
        self.last_drag_position = self.is_dragging.then_some(self.cursor_position).flatten();
        self.update_cursor();
    }

    fn handle_keyboard_input(&mut self, event_loop: &dyn ActiveEventLoop, event: &KeyEvent) {
        if event.state != ElementState::Pressed || event.repeat {
            return;
        }
        if let Some(action) = keyboard_action(event.physical_key, self.modifiers, self.fullscreen) {
            self.handle_keyboard_action(event_loop, action);
        }
    }

    fn handle_redraw(&mut self, event_loop: &dyn ActiveEventLoop) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        match gpu.render() {
            Err(error) => {
                self.fatal_error = Some(RuntimeError::Render(error));
                self.session.cancel_pending_presentation();
                event_loop.exit();
            }
            Ok(true) => {
                if let Some((path, opening)) = self.session.finish_presentation() {
                    let opening_time = opening.started.elapsed();
                    tracing::info!(
                        path = %path.display(),
                        purpose = "demand",
                        request = opening.request_kind(),
                        selection_generation = opening.generation,
                        direction = ?opening.direction,
                        cache_hit = opening.cache_hit,
                        width = opening.width,
                        height = opening.height,
                        decode_ms = milliseconds(opening.decode_time),
                        first_present_ms = milliseconds(opening_time),
                        "image opening completed"
                    );
                    if self.diagnostics {
                        let report = format!(
                            "  path: {}\n  request: {}\n  selection generation: {}\n  cache hit: {}\n  dimensions: {}x{}\n  decode: {:.2} ms\n  first presentation submission: {:.2} ms\n{}",
                            path.display(),
                            opening.request_kind(),
                            opening.generation,
                            opening.cache_hit,
                            opening.width,
                            opening.height,
                            milliseconds(opening.decode_time),
                            milliseconds(opening_time),
                            gpu.image_diagnostics_report(),
                        );
                        diagnostics::print_report("image-presented", &report);
                    }
                }
            }
            Ok(false) => {}
        }
    }

    fn shutdown(&mut self) {
        self.session.shutdown();
        if let Some(report) = self.current_image_finished_report("shutdown") {
            diagnostics::print_report("image-finished", &report);
        }
    }

    fn current_image_finished_report(&self, reason: &str) -> Option<String> {
        if !self.diagnostics || !self.session.current_image_was_presented() {
            return None;
        }
        let path = self.session.current_path()?;
        let gpu = self.gpu.as_ref()?;
        Some(format!(
            "  path: {}\n  reason: {reason}\n{}",
            path.display(),
            gpu.image_finished_diagnostics_report(),
        ))
    }
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn fullscreen_cursor_hide_deadline(
    fullscreen: FullscreenState,
    is_dragging: bool,
    now: Instant,
) -> Option<Instant> {
    (fullscreen == FullscreenState::Fullscreen && !is_dragging)
        .then_some(now + FULLSCREEN_CURSOR_HIDE_DELAY)
}

fn next_wake_deadline(
    prefetch_deadline: Option<Instant>,
    cursor_hide_deadline: Option<Instant>,
) -> Option<Instant> {
    [prefetch_deadline, cursor_hide_deadline]
        .into_iter()
        .flatten()
        .min()
}

#[derive(Debug)]
enum UserEvent {
    DecodeComplete(Box<DecodeCompletion>),
    FilePickerComplete(FilePickerResult),
    FolderNavigationComplete {
        generation: u64,
        direction: FolderDirection,
        source_path: PathBuf,
        result: Result<Option<PathBuf>, String>,
    },
    NeighborPathsComplete {
        generation: u64,
        anchor_key: ImageKey,
        result: Result<NeighborPaths, String>,
    },
    GpuFailure(gpu::GpuFailure),
    GpuWorkReady,
}

impl ApplicationHandler for Application {
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = WindowAttributes::default()
            .with_title(APPLICATION_NAME)
            .with_window_icon(Some(app_icon::window_icon()))
            .with_surface_size(LogicalSize::new(1024.0, 682.0));
        #[cfg(target_os = "linux")]
        let attributes = {
            let mut wayland_attributes =
                WindowAttributesWayland::default().with_name(APPLICATION_ID, APPLICATION_ID);
            if let Some(token) = event_loop.read_token_from_env() {
                startup_notify::reset_activation_token_env();
                wayland_attributes = wayland_attributes.with_activation_token(token);
            }
            attributes.with_platform_attributes(Box::new(wayland_attributes))
        };
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::<dyn Window>::from(window),
            Err(error) => {
                self.fatal_error = Some(RuntimeError::WindowCreation(error));
                event_loop.exit();
                return;
            }
        };

        let event_sender = self.event_sender.clone();
        let report_failure = Arc::new(move |failure| {
            event_sender.send(UserEvent::GpuFailure(failure));
        });
        let work_event_sender = self.event_sender.clone();
        let notify_work_ready = Arc::new(move || {
            work_event_sender.send(UserEvent::GpuWorkReady);
        });
        match gpu::initialize(
            &self.instance,
            Arc::clone(&window),
            self.output_mode,
            gpu::RenderingOptions {
                exposure_stops: 0.0,
                background: self.background,
            },
            self.diagnostics,
            report_failure,
            notify_work_ready,
            self.gpu_memory_limit_bytes,
        ) {
            Ok(mut gpu) => {
                if let Some(image) = self.session.take_decoded_image()
                    && let Err(error) = gpu.set_image(&image)
                {
                    self.fatal_error = Some(RuntimeError::ImageInstall(error));
                    self.session.cancel_pending_presentation();
                    event_loop.exit();
                    return;
                }
                if self.diagnostics {
                    diagnostics::print_report("startup", &gpu.startup_diagnostics_report());
                }
                window.request_redraw();
                self.gpu = Some(gpu);
                self.window = Some(window);
                self.update_window_title();
                self.refresh_metadata_overlay();
                self.update_cursor();
                if let Some(window) = self.window.as_ref() {
                    window.set_cursor_visible(self.cursor_visibility == CursorVisibility::Visible);
                }
            }
            Err(error) => {
                self.fatal_error = Some(RuntimeError::GpuInitialization(error));
                event_loop.exit();
            }
        }
    }

    fn proxy_wake_up(&mut self, event_loop: &dyn ActiveEventLoop) {
        while let Ok(event) = self.event_receiver.try_recv() {
            self.handle_user_event(event_loop, event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        let now = Instant::now();
        if self
            .session
            .prefetch_deadline()
            .is_some_and(|deadline| now >= deadline)
        {
            self.start_neighbor_lookup();
        }
        self.hide_fullscreen_cursor_if_due(now);

        let prefetch_deadline = self.session.prefetch_deadline();
        match next_wake_deadline(prefetch_deadline, self.cursor_hide_deadline) {
            Some(deadline) => event_loop.set_control_flow(ControlFlow::WaitUntil(deadline)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::SurfaceResized(size) => {
                self.handle_surface_resized(event_loop, size);
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.handle_scale_factor_changed(event_loop, scale_factor);
            }
            WindowEvent::Moved(_) => self.handle_window_moved(event_loop),
            WindowEvent::Focused(focused) => self.handle_focus_changed(event_loop, focused),
            WindowEvent::DragEntered { id, .. } => self.begin_file_drop(event_loop, id),
            WindowEvent::DragDropped { id, .. } => self.handle_file_dropped(id),
            WindowEvent::DragLeft { id } => self.handle_file_drag_left(id),
            WindowEvent::DataTransferReceived { id, serial, value } => {
                self.handle_data_transfer_received(id, serial, value.as_ref());
            }
            WindowEvent::PointerEntered { position, .. } => self.handle_pointer_entered(position),
            WindowEvent::PointerLeft { .. } => self.handle_pointer_departure(),
            WindowEvent::PointerMoved { position, .. } => self.handle_pointer_moved(position),
            WindowEvent::PointerButton { state, button, .. } => {
                self.handle_pointer_button(state, &button);
            }
            WindowEvent::MouseWheel { delta, .. } => self.handle_mouse_wheel(delta),
            WindowEvent::PinchGesture { delta, .. } => self.handle_pinch_gesture(delta),
            WindowEvent::PanGesture { delta, .. } => self.handle_pan_gesture(delta),
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard_input(event_loop, &event);
            }
            WindowEvent::RedrawRequested => self.handle_redraw(event_loop),
            _ => {}
        }
    }
}

pub(super) fn run(cli: &Cli) -> Result<(), AppError> {
    #[cfg(target_os = "linux")]
    require_native_wayland()?;
    let decoded_cache_bytes = cli.cache.map_or_else(automatic_decoded_cache_bytes, |mib| {
        mib.saturating_mul(1024 * 1024)
    });
    log_startup_options(cli, decoded_cache_bytes);

    let mut event_loop_builder = EventLoop::builder();
    #[cfg(target_os = "linux")]
    event_loop_builder.with_wayland();
    let mut event_loop = event_loop_builder.build()?;

    let mut gpu_descriptor = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(
        event_loop.owned_display_handle(),
    ));
    gpu_descriptor.backends = gpu::native_backends();

    let (event_sender, event_receiver) = mpsc::channel();
    let event_sender = UserEventSender {
        sender: event_sender,
        proxy: event_loop.create_proxy(),
    };
    let decode_limits = DecodeLimits::default();
    let decode_event_sender = event_sender.clone();
    let decode_coordinator = DecodeCoordinator::spawn(decode_limits, move |completion| {
        decode_event_sender.send(UserEvent::DecodeComplete(Box::new(completion)));
    })
    .map_err(RuntimeError::DecodeCoordinator)?;
    let mut application = Application {
        background: cli.background,
        cursor_hide_deadline: None,
        cursor_position: None,
        cursor_visibility: CursorVisibility::Visible,
        diagnostics: cli.diagnostics,
        fatal_error: None,
        event_receiver,
        event_sender,
        file_picker_state: FilePickerState::Closed,
        fullscreen: FullscreenState::Windowed,
        gpu: None,
        instance: wgpu::Instance::new(gpu_descriptor),
        is_dragging: false,
        last_drag_position: None,
        last_primary_click: None,
        metadata_visible: false,
        modifiers: ModifiersState::empty(),
        output_mode: cli.output,
        pending_drop: None,
        session: ImageSession::new(decoded_cache_bytes, decode_coordinator),
        gpu_memory_limit_bytes: cli.gpu_memory.saturating_mul(1024 * 1024),
        window: None,
    };

    if let Some(path) = cli.image.clone() {
        application.select_path(path, None);
    }

    let event_loop_result = event_loop.run_app_on_demand(&mut application);
    application.shutdown();
    event_loop_result?;
    application
        .fatal_error
        .map_or(Ok(()), |error| Err(error.into()))
}

fn log_startup_options(cli: &Cli, decoded_cache_bytes: u64) {
    let image = cli
        .image
        .as_ref()
        .map_or("(none)", |path| path.to_str().unwrap_or("(non-UTF-8 path)"));
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        image,
        output = cli.output.as_str(),
        background = cli.background.as_str(),
        hdr_reference_white_nits = HDR_REFERENCE_WHITE_NITS,
        decoded_cache_mib = decoded_cache_bytes / (1024 * 1024),
        decoded_cache_automatic = cli.cache.is_none(),
        gpu_memory_mib = cli.gpu_memory,
        diagnostics = cli.diagnostics,
        asynchronous_image_loading = cli.image.is_some(),
        "starting xl-view"
    );
}

#[cfg(target_os = "linux")]
fn require_native_wayland() -> Result<(), AppError> {
    let display_name = std::env::var_os("WAYLAND_DISPLAY");
    let inherited_socket = std::env::var_os("WAYLAND_SOCKET");

    if has_wayland_endpoint(display_name.as_deref(), inherited_socket.as_deref()) {
        Ok(())
    } else {
        Err(AppError::NoNativeWayland)
    }
}

#[cfg(target_os = "linux")]
fn has_wayland_endpoint(display_name: Option<&OsStr>, inherited_socket: Option<&OsStr>) -> bool {
    [display_name, inherited_socket]
        .into_iter()
        .flatten()
        .any(|value| !value.is_empty())
}

#[cfg(test)]
mod tests;
