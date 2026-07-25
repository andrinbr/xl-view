use std::time::{Duration, Instant};

use winit::cursor::CursorIcon;
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

pub(super) fn is_double_click(previous: Instant, current: Instant) -> bool {
    const MAX_INTERVAL: Duration = Duration::from_millis(400);
    current.saturating_duration_since(previous) <= MAX_INTERVAL
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
        (true, false) => CursorIcon::Grab,
        (false, _) => CursorIcon::Default,
    }
}
