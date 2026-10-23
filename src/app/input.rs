use std::time::{Duration, Instant};

use winit::cursor::CursorIcon;
use winit::dpi::PhysicalPosition;
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};

pub(super) fn keyboard_action(
    key: PhysicalKey,
    modifiers: ModifiersState,
    fullscreen: FullscreenState,
) -> Option<KeyboardAction> {
    // Shift+O used to cycle output modes. It now does nothing instead of
    // falling through to Open.
    if modifiers.shift_key() && key == PhysicalKey::Code(KeyCode::KeyO) {
        return None;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyboard_bindings_match_the_interaction_contract() {
        let bindings = [
            (KeyCode::KeyF, KeyboardAction::Fit),
            (KeyCode::Digit1, KeyboardAction::OneToOne),
            (KeyCode::Numpad1, KeyboardAction::OneToOne),
            (KeyCode::Equal, KeyboardAction::ZoomIn),
            (KeyCode::NumpadAdd, KeyboardAction::ZoomIn),
            (KeyCode::Minus, KeyboardAction::ZoomOut),
            (KeyCode::NumpadSubtract, KeyboardAction::ZoomOut),
            (KeyCode::KeyO, KeyboardAction::Open),
            (KeyCode::KeyQ, KeyboardAction::Quit),
            (KeyCode::BracketLeft, KeyboardAction::ExposureDown),
            (KeyCode::BracketRight, KeyboardAction::ExposureUp),
            (KeyCode::KeyR, KeyboardAction::ResetViewAndExposure),
            (KeyCode::KeyB, KeyboardAction::CycleBackground),
            (KeyCode::KeyI, KeyboardAction::ToggleMetadata),
            (KeyCode::F11, KeyboardAction::ToggleFullscreen),
            (KeyCode::Enter, KeyboardAction::ToggleFullscreen),
            (KeyCode::KeyA, KeyboardAction::PanLeft),
            (KeyCode::KeyD, KeyboardAction::PanRight),
            (KeyCode::KeyW, KeyboardAction::PanUp),
            (KeyCode::KeyS, KeyboardAction::PanDown),
            (KeyCode::ArrowLeft, KeyboardAction::PreviousImage),
            (KeyCode::ArrowRight, KeyboardAction::NextImage),
        ];
        for (key, expected) in bindings {
            assert_eq!(
                keyboard_action(
                    PhysicalKey::Code(key),
                    ModifiersState::empty(),
                    FullscreenState::Windowed,
                ),
                Some(expected),
            );
        }

        assert_eq!(
            keyboard_action(
                PhysicalKey::Code(KeyCode::KeyO),
                ModifiersState::CONTROL,
                FullscreenState::Windowed,
            ),
            Some(KeyboardAction::Open),
        );
        assert_eq!(
            keyboard_action(
                PhysicalKey::Code(KeyCode::KeyQ),
                ModifiersState::CONTROL,
                FullscreenState::Windowed,
            ),
            Some(KeyboardAction::Quit),
        );
        assert_eq!(
            keyboard_action(
                PhysicalKey::Code(KeyCode::Escape),
                ModifiersState::empty(),
                FullscreenState::Fullscreen,
            ),
            Some(KeyboardAction::LeaveFullscreen),
        );
        assert_eq!(
            keyboard_action(
                PhysicalKey::Code(KeyCode::Escape),
                ModifiersState::empty(),
                FullscreenState::Windowed,
            ),
            None,
        );
    }

    #[test]
    fn fullscreen_requested_state_toggles_without_waiting_for_the_compositor() {
        assert_eq!(
            FullscreenState::Windowed.toggled(),
            FullscreenState::Fullscreen
        );
        assert_eq!(
            FullscreenState::Fullscreen.toggled(),
            FullscreenState::Windowed
        );
    }

    #[test]
    fn primary_click_tracker_uses_a_300_millisecond_interval() {
        let start = Instant::now();
        for (interval, is_double_click) in [(299, true), (300, true), (301, false)] {
            let mut tracker = PrimaryClickTracker::default();
            tracker.press(start, Some(PhysicalPosition::new(0.0, 0.0)), 1.0);
            assert!(!tracker.release());
            tracker.press(
                start + Duration::from_millis(interval),
                Some(PhysicalPosition::new(0.0, 0.0)),
                1.0,
            );
            assert_eq!(tracker.release(), is_double_click);
        }
    }

    #[test]
    fn primary_click_tracker_uses_logical_distance_from_the_press() {
        let start = Instant::now();
        let origin = PhysicalPosition::new(100.0, 100.0);
        let mut tracker = PrimaryClickTracker::default();
        tracker.press(start, Some(origin), 2.0);
        assert!(!tracker.release());

        tracker.press(start + Duration::from_millis(250), Some(origin), 2.0);
        tracker.move_to(PhysicalPosition::new(106.0, 107.0));
        assert!(tracker.release());
    }

    #[test]
    fn hovering_between_clicks_does_not_cancel_the_sequence() {
        let start = Instant::now();
        let origin = PhysicalPosition::new(100.0, 100.0);
        let mut tracker = PrimaryClickTracker::default();
        tracker.press(start, Some(origin), 1.0);
        assert!(!tracker.release());

        tracker.move_to(PhysicalPosition::new(500.0, 500.0));
        tracker.press(start + Duration::from_millis(250), Some(origin), 1.0);
        assert!(tracker.release());
    }

    #[test]
    fn unmatched_release_cancels_the_click_sequence() {
        let start = Instant::now();
        let origin = PhysicalPosition::new(100.0, 100.0);
        let mut tracker = PrimaryClickTracker::default();
        tracker.press(start, Some(origin), 1.0);
        assert!(!tracker.release());

        assert!(!tracker.release());
        tracker.press(start + Duration::from_millis(250), Some(origin), 1.0);
        assert!(!tracker.release());
    }

    #[test]
    fn drag_cancels_a_pending_double_click() {
        let start = Instant::now();
        let origin = PhysicalPosition::new(100.0, 100.0);
        let mut tracker = PrimaryClickTracker::default();
        tracker.press(start, Some(origin), 1.0);
        assert!(!tracker.release());

        tracker.press(start + Duration::from_millis(250), Some(origin), 1.0);
        tracker.move_to(PhysicalPosition::new(103.0, 104.0));
        assert!(!tracker.release());

        tracker.press(start + Duration::from_millis(275), Some(origin), 1.0);
        assert!(!tracker.release());
    }

    #[test]
    fn movement_without_a_press_origin_cancels_the_click_sequence() {
        let start = Instant::now();
        let origin = PhysicalPosition::new(100.0, 100.0);
        let mut tracker = PrimaryClickTracker::default();
        tracker.press(start, Some(origin), 1.0);
        assert!(!tracker.release());

        tracker.press(start + Duration::from_millis(250), None, 1.0);
        tracker.move_to(origin);
        assert!(!tracker.release());

        tracker.press(start + Duration::from_millis(275), Some(origin), 1.0);
        assert!(!tracker.release());
    }

    #[test]
    fn image_cursor_shows_pan_availability_and_drag_state() {
        assert_eq!(image_cursor(false, false), CursorIcon::Default);
        assert_eq!(image_cursor(false, true), CursorIcon::Default);
        #[cfg(target_os = "windows")]
        {
            assert_eq!(image_cursor(true, false), CursorIcon::Default);
            assert_eq!(image_cursor(true, true), CursorIcon::Grabbing);
        }
        #[cfg(not(target_os = "windows"))]
        {
            assert_eq!(image_cursor(true, false), CursorIcon::Grab);
            assert_eq!(image_cursor(true, true), CursorIcon::Grabbing);
        }
    }
}
