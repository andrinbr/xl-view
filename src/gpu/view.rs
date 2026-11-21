use winit::dpi::{PhysicalPosition, PhysicalSize};

const MIN_SCALE: f64 = 1.0 / 64.0;
const MAX_SCALE: f64 = 64.0;
const MAX_ZOOM_FROM_FIT: f64 = 16.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewMode {
    Fit,
    OneToOne,
    Manual,
}

/// Visible image region in image pixels, clamped to the image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageBounds {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ViewTransform {
    image: PhysicalSize<u32>,
    viewport: PhysicalSize<u32>,
    center: PhysicalPosition<f64>,
    scale: f64,
    mode: ViewMode,
}

impl ViewTransform {
    pub fn fit(image: PhysicalSize<u32>, viewport: PhysicalSize<u32>) -> Self {
        let mut view = Self {
            image,
            viewport,
            center: PhysicalPosition::new(
                f64::from(image.width) / 2.0,
                f64::from(image.height) / 2.0,
            ),
            scale: 1.0,
            mode: ViewMode::Fit,
        };
        view.refit();
        view
    }

    /// The center to render from. At scale 1.0 it is snapped so image pixel
    /// centers land on screen pixel centers, and bilinear sampling copies
    /// pixels exactly instead of averaging neighbors.
    pub fn center(self) -> PhysicalPosition<f64> {
        if self.scale.to_bits() != 1.0_f64.to_bits() {
            return self.center;
        }
        let snap = |center: f64, viewport: u32| {
            let half_viewport = f64::from(viewport) / 2.0;
            (center - half_viewport).round() + half_viewport
        };
        PhysicalPosition::new(
            snap(self.center.x, self.viewport.width),
            snap(self.center.y, self.viewport.height),
        )
    }

    pub fn scale(self) -> f64 {
        self.scale
    }

    pub fn visible_image_bounds(self) -> ImageBounds {
        let half_width = f64::from(self.viewport.width) / self.scale / 2.0;
        let half_height = f64::from(self.viewport.height) / self.scale / 2.0;
        ImageBounds {
            left: (self.center.x - half_width).max(0.0),
            top: (self.center.y - half_height).max(0.0),
            right: (self.center.x + half_width).min(f64::from(self.image.width)),
            bottom: (self.center.y + half_height).min(f64::from(self.image.height)),
        }
    }

    pub fn set_viewport(&mut self, viewport: PhysicalSize<u32>) {
        self.viewport = viewport;
        if self.mode == ViewMode::Fit {
            self.refit();
        } else {
            self.clamp_center();
        }
    }

    pub fn set_fit(&mut self) {
        self.mode = ViewMode::Fit;
        self.refit();
    }

    pub fn set_one_to_one(&mut self, scale_factor: f64) {
        self.mode = ViewMode::OneToOne;
        self.scale = scale_factor.clamp(MIN_SCALE, MAX_SCALE);
        self.clamp_center();
    }

    pub fn set_scale_factor(&mut self, scale_factor: f64) {
        if self.mode == ViewMode::OneToOne {
            self.scale = scale_factor.clamp(MIN_SCALE, MAX_SCALE);
            self.clamp_center();
        }
    }

    /// Returns whether this zoom snapped to the fit stop.
    pub fn zoom_at(&mut self, cursor: PhysicalPosition<f64>, factor: f64) -> bool {
        if !factor.is_finite() || factor <= 0.0 {
            return false;
        }
        let old_scale = self.scale;
        let fit_scale = self.fit_scale();
        let requested_scale = old_scale * factor;
        // Make fit an exact zoom stop instead of skipping across its arbitrary scale.
        if (old_scale > fit_scale && requested_scale <= fit_scale)
            || (old_scale < fit_scale && requested_scale >= fit_scale)
        {
            self.set_fit();
            return true;
        }
        // Anchor on the rendered center, which is snapped at 1:1, so the
        // image point shown under the cursor stays there.
        let rendered_center = self.center();
        let image_at_cursor = PhysicalPosition::new(
            rendered_center.x + (cursor.x - f64::from(self.viewport.width) / 2.0) / old_scale,
            rendered_center.y + (cursor.y - f64::from(self.viewport.height) / 2.0) / old_scale,
        );
        // Fit can already exceed the usual absolute limit for tiny images. Base the manual limit
        // on both values so the first zoom-in never makes a fitted image smaller.
        let maximum_scale = self.maximum_manual_scale().max(old_scale);
        // Keep fit reachable when a very large image needs a scale below the usual minimum.
        self.scale = requested_scale.clamp(MIN_SCALE.min(fit_scale), maximum_scale);
        self.center = PhysicalPosition::new(
            image_at_cursor.x - (cursor.x - f64::from(self.viewport.width) / 2.0) / self.scale,
            image_at_cursor.y - (cursor.y - f64::from(self.viewport.height) / 2.0) / self.scale,
        );
        self.mode = ViewMode::Manual;
        self.clamp_center();
        false
    }

    pub fn pan_by(&mut self, delta_x: f64, delta_y: f64) {
        if !delta_x.is_finite() || !delta_y.is_finite() {
            return;
        }
        self.mode = ViewMode::Manual;
        self.center.x -= delta_x / self.scale;
        self.center.y -= delta_y / self.scale;
        self.clamp_center();
    }

    fn refit(&mut self) {
        self.scale = self.fit_scale();
        self.center = PhysicalPosition::new(
            f64::from(self.image.width) / 2.0,
            f64::from(self.image.height) / 2.0,
        );
    }

    fn fit_scale(&self) -> f64 {
        let width_scale =
            f64::from(self.viewport.width.max(1)) / f64::from(self.image.width.max(1));
        let height_scale =
            f64::from(self.viewport.height.max(1)) / f64::from(self.image.height.max(1));
        width_scale.min(height_scale).max(f64::MIN_POSITIVE)
    }

    fn maximum_manual_scale(&self) -> f64 {
        MAX_SCALE.max(self.fit_scale() * MAX_ZOOM_FROM_FIT)
    }

    fn clamp_center(&mut self) {
        self.center.x = clamp_axis(
            self.center.x,
            self.image.width,
            self.viewport.width,
            self.scale,
        );
        self.center.y = clamp_axis(
            self.center.y,
            self.image.height,
            self.viewport.height,
            self.scale,
        );
    }
}

fn clamp_axis(center: f64, image: u32, viewport: u32, scale: f64) -> f64 {
    let image = f64::from(image);
    let visible = f64::from(viewport) / scale;
    if visible >= image {
        image / 2.0
    } else {
        center.clamp(visible / 2.0, image - visible / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    #[test]
    fn fit_letterboxes_without_cropping() {
        let view = ViewTransform::fit(PhysicalSize::new(4000, 2000), PhysicalSize::new(1000, 1000));
        assert_near(view.scale(), 0.25);
        assert_eq!(view.center(), PhysicalPosition::new(2000.0, 1000.0));
    }

    #[test]
    fn one_to_one_puts_image_pixels_on_screen_pixels() {
        // Screen pixel x samples center - viewport / 2 + x + 0.5, which is an
        // image pixel center only when center - viewport / 2 is whole.
        for (viewport, expected) in [
            (1023, PhysicalPosition::new(2000.5, 1500.5)),
            (1024, PhysicalPosition::new(2000.0, 1500.0)),
        ] {
            let mut view = ViewTransform::fit(
                PhysicalSize::new(4000, 3000),
                PhysicalSize::new(viewport, viewport),
            );
            view.set_one_to_one(1.0);
            assert_eq!(view.center(), expected, "{viewport}");
        }
    }

    #[test]
    fn one_to_one_pans_in_whole_pixels_but_keeps_sub_pixel_motion() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(4000, 3000), PhysicalSize::new(1024, 1024));
        view.set_one_to_one(1.0);
        view.pan_by(-0.3, 0.0);
        assert_eq!(view.center(), PhysicalPosition::new(2000.0, 1500.0));
        for _ in 0..3 {
            view.pan_by(-0.3, 0.0);
        }
        assert_eq!(view.center(), PhysicalPosition::new(2001.0, 1500.0));

        view.set_one_to_one(2.0);
        view.pan_by(-0.6, 0.0);
        assert_near(view.center().x, 2001.5);
    }

    #[test]
    fn zooming_out_of_one_to_one_keeps_the_shown_point_under_the_cursor() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(4000, 3000), PhysicalSize::new(1023, 767));
        view.set_one_to_one(1.0);
        // The odd viewport snaps the rendered center half a pixel.
        assert_eq!(view.center(), PhysicalPosition::new(2000.5, 1500.5));

        view.zoom_at(PhysicalPosition::new(511.5, 383.5), 2.0);

        assert_near(view.scale(), 2.0);
        assert_eq!(view.center(), PhysicalPosition::new(2000.5, 1500.5));
    }

    #[test]
    fn cursor_centered_zoom_preserves_image_point() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(1000, 1000), PhysicalSize::new(1000, 1000));
        view.zoom_at(PhysicalPosition::new(750.0, 500.0), 2.0);
        assert_near(view.scale(), 2.0);
        assert_eq!(view.center(), PhysicalPosition::new(625.0, 500.0));
    }

    #[test]
    fn zoom_out_from_one_to_one_stops_at_fit() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(3000, 2000), PhysicalSize::new(1000, 1000));
        view.set_one_to_one(1.0);

        for _ in 0..4 {
            view.zoom_at(PhysicalPosition::new(500.0, 500.0), 0.8);
        }
        assert_near(view.scale(), 0.4096);

        assert!(view.zoom_at(PhysicalPosition::new(500.0, 500.0), 0.8));

        assert_near(view.scale(), 1.0 / 3.0);
        assert_eq!(view.center(), PhysicalPosition::new(1500.0, 1000.0));
        assert_eq!(view.mode, ViewMode::Fit);
    }

    #[test]
    fn zoom_in_stops_at_fit_when_starting_below_it() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(2000, 1000), PhysicalSize::new(1000, 1000));
        view.zoom_at(PhysicalPosition::new(500.0, 500.0), 0.5);
        assert_near(view.scale(), 0.25);

        assert!(view.zoom_at(PhysicalPosition::new(500.0, 500.0), 3.0));

        assert_near(view.scale(), 0.5);
        assert_eq!(view.mode, ViewMode::Fit);
    }

    #[test]
    fn zoom_can_continue_past_the_fit_stop() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(3000, 2000), PhysicalSize::new(1000, 1000));
        view.set_one_to_one(1.0);
        assert!(view.zoom_at(PhysicalPosition::new(500.0, 500.0), 0.3));
        assert_eq!(view.mode, ViewMode::Fit);

        assert!(!view.zoom_at(PhysicalPosition::new(500.0, 500.0), 0.8));

        assert_near(view.scale(), (1.0 / 3.0) * 0.8);
        assert_eq!(view.mode, ViewMode::Manual);
    }

    #[test]
    fn fit_stop_remains_reachable_below_the_usual_manual_minimum() {
        let mut view = ViewTransform::fit(
            PhysicalSize::new(100_000, 1000),
            PhysicalSize::new(1000, 1000),
        );
        view.set_one_to_one(1.0);

        for _ in 0..32 {
            view.zoom_at(PhysicalPosition::new(500.0, 500.0), 0.8);
            if view.mode == ViewMode::Fit {
                break;
            }
        }

        assert_near(view.scale(), 0.01);
        assert_eq!(view.mode, ViewMode::Fit);
    }

    #[test]
    fn tiny_fitted_image_zooms_in_without_shrinking() {
        let mut view = ViewTransform::fit(PhysicalSize::new(2, 3), PhysicalSize::new(600, 600));
        assert_near(view.scale(), 200.0);
        view.zoom_at(PhysicalPosition::new(300.0, 300.0), 1.25);
        assert_near(view.scale(), 250.0);
    }

    #[test]
    fn tiny_image_manual_zoom_is_limited_relative_to_fit() {
        let mut view = ViewTransform::fit(PhysicalSize::new(2, 3), PhysicalSize::new(600, 600));
        view.zoom_at(PhysicalPosition::new(300.0, 300.0), 100.0);
        assert_near(view.scale(), 3_200.0);
    }

    #[test]
    fn ordinary_image_keeps_the_absolute_manual_zoom_limit() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(1000, 1000), PhysicalSize::new(1000, 1000));
        view.zoom_at(PhysicalPosition::new(500.0, 500.0), 100.0);
        assert_near(view.scale(), MAX_SCALE);
    }

    #[test]
    fn zoom_in_never_shrinks_after_a_manual_viewport_change() {
        let mut view = ViewTransform::fit(PhysicalSize::new(2, 3), PhysicalSize::new(600, 600));
        view.zoom_at(PhysicalPosition::new(300.0, 300.0), 10.0);
        assert_near(view.scale(), 2_000.0);
        view.set_viewport(PhysicalSize::new(60, 60));
        view.zoom_at(PhysicalPosition::new(30.0, 30.0), 2.0);
        assert_near(view.scale(), 2_000.0);
    }

    #[test]
    fn pan_clamps_at_edges_but_keeps_them_visible() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(2000, 1000), PhysicalSize::new(1000, 1000));
        view.set_one_to_one(1.0);
        view.pan_by(10_000.0, 0.0);
        assert_near(view.center().x, 500.0);
        view.pan_by(-10_000.0, 0.0);
        assert_near(view.center().x, 1500.0);
    }

    #[test]
    fn non_finite_pan_is_ignored() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(2000, 1000), PhysicalSize::new(1000, 1000));
        view.set_one_to_one(1.0);
        let center = view.center();
        let mode = view.mode;

        for (delta_x, delta_y) in [
            (f64::NAN, 0.0),
            (0.0, f64::NAN),
            (f64::INFINITY, 0.0),
            (0.0, f64::NEG_INFINITY),
        ] {
            view.pan_by(delta_x, delta_y);
            assert_eq!(view.center(), center);
            assert_eq!(view.mode, mode);
        }
    }

    #[test]
    fn one_to_one_uses_logical_pixel_scale() {
        let mut view = ViewTransform::fit(PhysicalSize::new(100, 100), PhysicalSize::new(800, 600));
        view.set_one_to_one(2.0);
        assert_near(view.scale(), 2.0);
        view.set_scale_factor(1.5);
        assert_near(view.scale(), 1.5);
    }

    #[test]
    fn one_to_one_preserves_the_current_image_center() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(4000, 3000), PhysicalSize::new(1000, 1000));
        view.set_one_to_one(1.0);
        view.pan_by(-400.0, -300.0);
        let center = view.center();

        view.set_one_to_one(2.0);

        assert_near(view.scale(), 2.0);
        assert_eq!(view.center(), center);
    }

    #[test]
    fn visible_bounds_are_clamped_to_the_image() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(2000, 1000), PhysicalSize::new(1000, 1000));
        assert_eq!(
            view.visible_image_bounds(),
            ImageBounds {
                left: 0.0,
                top: 0.0,
                right: 2000.0,
                bottom: 1000.0,
            }
        );
        view.set_one_to_one(1.0);
        assert_eq!(
            view.visible_image_bounds(),
            ImageBounds {
                left: 500.0,
                top: 0.0,
                right: 1500.0,
                bottom: 1000.0,
            }
        );
    }

    #[test]
    fn manual_zoom_survives_monitor_scale_changes() {
        let mut view = ViewTransform::fit(PhysicalSize::new(100, 100), PhysicalSize::new(800, 600));
        view.zoom_at(PhysicalPosition::new(400.0, 300.0), 2.0);
        let scale = view.scale();
        view.set_scale_factor(1.5);
        assert_near(view.scale(), scale);
    }
}
