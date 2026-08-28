use std::time::{Duration, Instant};

use winit::cursor::CursorIcon;
use winit::dpi::PhysicalPosition;
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};

pub(super) fn keyboard_action(
    key: PhysicalKey,
    modifiers: ModifiersState,
    fullscreen: FullscreenState,
) -> Option<KeyboardAction> {
    if modifiers.shift_key() && key == PhysicalKey::Code(KeyCode::KeyO) {
        return None;
    }

    if modifiers.control_key() {
        match key {
            PhysicalKey::Code(KeyCode::KeyQ) => return Some(KeyboardAction::Quit),
            PhysicalKey::Code(KeyCode::KeyO) => return Some(KeyboardAction::Open),
            _ => {}
        }
    }

    match key {
        PhysicalKey::Code(KeyCode::KeyF) => Some(KeyboardAction::Fit),
        PhysicalKey::Code(KeyCode::Digit1 | KeyCode::Numpad1) => Some(KeyboardAction::OneToOne),
        PhysicalKey::Code(KeyCode::Equal | KeyCode::NumpadAdd) => Some(KeyboardAction::ZoomIn),
        PhysicalKey::Code(KeyCode::Minus | KeyCode::NumpadSubtract) => {
            Some(KeyboardAction::ZoomOut)
        }
        PhysicalKey::Code(KeyCode::KeyO) => Some(KeyboardAction::Open),
        PhysicalKey::Code(KeyCode::KeyQ) => Some(KeyboardAction::Quit),
        PhysicalKey::Code(KeyCode::BracketLeft) => Some(KeyboardAction::ExposureDown),
        PhysicalKey::Code(KeyCode::BracketRight) => Some(KeyboardAction::ExposureUp),
        PhysicalKey::Code(KeyCode::KeyR) => Some(KeyboardAction::ResetViewAndExposure),
        PhysicalKey::Code(KeyCode::KeyB) => Some(KeyboardAction::CycleBackground),
        PhysicalKey::Code(KeyCode::KeyI) => Some(KeyboardAction::ToggleMetadata),
        PhysicalKey::Code(KeyCode::F11 | KeyCode::Enter) => Some(KeyboardAction::ToggleFullscreen),
        PhysicalKey::Code(KeyCode::Escape) if fullscreen == FullscreenState::Fullscreen => {
            Some(KeyboardAction::LeaveFullscreen)
        }
        PhysicalKey::Code(KeyCode::KeyA) => Some(KeyboardAction::PanLeft),
        PhysicalKey::Code(KeyCode::KeyD) => Some(KeyboardAction::PanRight),
        PhysicalKey::Code(KeyCode::KeyW) => Some(KeyboardAction::PanUp),
        PhysicalKey::Code(KeyCode::KeyS) => Some(KeyboardAction::PanDown),
        PhysicalKey::Code(KeyCode::ArrowLeft) => Some(KeyboardAction::PreviousImage),
        PhysicalKey::Code(KeyCode::ArrowRight) => Some(KeyboardAction::NextImage),
        _ => None,
    }
}

fn is_double_click(previous: Instant, current: Instant) -> bool {
    const MAX_INTERVAL: Duration = Duration::from_millis(300);
    current.saturating_duration_since(previous) <= MAX_INTERVAL
}

fn crossed_drag_threshold(
    origin: PhysicalPosition<f64>,
    current: PhysicalPosition<f64>,
    scale_factor: f64,
) -> bool {
    const DRAG_THRESHOLD_LOGICAL: f64 = 5.0;
    let delta_x = current.x - origin.x;
    let delta_y = current.y - origin.y;
    // Winit positions are physical, so scale the logical drag slop for HiDPI.
    let physical_threshold = DRAG_THRESHOLD_LOGICAL * scale_factor;
    delta_x.mul_add(delta_x, delta_y * delta_y) >= physical_threshold * physical_threshold
}

#[derive(Clone, Copy, Debug)]
struct PrimaryPress {
    instant: Instant,
    position: Option<PhysicalPosition<f64>>,
    scale_factor: f64,
}

// Keeps a completed first click while deciding whether the next press is a click or drag.
#[derive(Debug, Default)]
pub(super) struct PrimaryClickTracker {
    last_click: Option<Instant>,
    press: Option<PrimaryPress>,
}

impl PrimaryClickTracker {
    pub(super) fn press(
        &mut self,
        instant: Instant,
        position: Option<PhysicalPosition<f64>>,
        scale_factor: f64,
    ) {
        self.press = Some(PrimaryPress {
            instant,
            position,
            scale_factor,
        });
    }

    pub(super) fn move_to(&mut self, position: PhysicalPosition<f64>) {
        // Without an origin, any movement cancels rather than allowing a drag as a click.
        let crossed_threshold = self.press.is_some_and(|press| {
            press
                .position
                .is_none_or(|origin| crossed_drag_threshold(origin, position, press.scale_factor))
        });
        if crossed_threshold {
            self.cancel();
        }
    }

    pub(super) fn release(&mut self) -> bool {
        // Only a press that survived drag detection can complete the click sequence.
        let Some(press) = self.press.take() else {
            self.cancel();
            return false;
        };
        if self
            .last_click
            .is_some_and(|previous| is_double_click(previous, press.instant))
        {
            self.last_click = None;
            true
        } else {
            self.last_click = Some(press.instant);
            false
        }
    }

    pub(super) fn cancel(&mut self) {
        self.last_click = None;
        self.press = None;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FullscreenState {
    Windowed,
    Fullscreen,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KeyboardAction {
    Quit,
    Open,
    Fit,
    OneToOne,
    ZoomIn,
    ZoomOut,
    ExposureDown,
    ExposureUp,
    ResetViewAndExposure,
    CycleBackground,
    ToggleMetadata,
    ToggleFullscreen,
    LeaveFullscreen,
    PanLeft,
    PanRight,
    PanUp,
    PanDown,
    PreviousImage,
    NextImage,
}

impl FullscreenState {
    pub(super) const fn toggled(self) -> Self {
        match self {
            Self::Windowed => Self::Fullscreen,
            Self::Fullscreen => Self::Windowed,
        }
    }
}

pub(super) fn image_cursor(has_image: bool, is_dragging: bool) -> CursorIcon {
    match (has_image, is_dragging) {
        (true, true) => CursorIcon::Grabbing,
        // Win32 has no predefined open/closed-hand cursors. Winit consequently
        // maps Grab to IDC_SIZEALL, so keep the normal arrow at rest.
        // https://github.com/rust-windowing/winit/issues/1043
        #[cfg(target_os = "windows")]
        (true, false) => CursorIcon::Default,
        #[cfg(not(target_os = "windows"))]
        (true, false) => CursorIcon::Grab,
        (false, _) => CursorIcon::Default,
    }
}
