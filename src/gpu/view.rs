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

    pub fn center(self) -> PhysicalPosition<f64> {
        self.center
    }

    pub fn scale(self) -> f64 {
        self.scale
    }

    pub fn visible_image_bounds(self) -> (f64, f64, f64, f64) {
        let half_width = f64::from(self.viewport.width) / self.scale / 2.0;
        let half_height = f64::from(self.viewport.height) / self.scale / 2.0;
        (
            (self.center.x - half_width).max(0.0),
            (self.center.y - half_height).max(0.0),
            (self.center.x + half_width).min(f64::from(self.image.width)),
            (self.center.y + half_height).min(f64::from(self.image.height)),
        )
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

    pub fn zoom_at(&mut self, cursor: PhysicalPosition<f64>, factor: f64) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let old_scale = self.scale;
        let image_at_cursor = PhysicalPosition::new(
            self.center.x + (cursor.x - f64::from(self.viewport.width) / 2.0) / old_scale,
            self.center.y + (cursor.y - f64::from(self.viewport.height) / 2.0) / old_scale,
        );
        // Fit can already exceed the usual absolute limit for tiny images. Base the manual limit
        // on both values so the first zoom-in never makes a fitted image smaller.
        let maximum_scale = self.maximum_manual_scale().max(old_scale);
        self.scale = (old_scale * factor).clamp(MIN_SCALE, maximum_scale);
        self.center = PhysicalPosition::new(
            image_at_cursor.x - (cursor.x - f64::from(self.viewport.width) / 2.0) / self.scale,
            image_at_cursor.y - (cursor.y - f64::from(self.viewport.height) / 2.0) / self.scale,
        );
        self.mode = ViewMode::Manual;
        self.clamp_center();
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
    fn cursor_centered_zoom_preserves_image_point() {
        let mut view =
            ViewTransform::fit(PhysicalSize::new(1000, 1000), PhysicalSize::new(1000, 1000));
        view.zoom_at(PhysicalPosition::new(750.0, 500.0), 2.0);
        assert_near(view.scale(), 2.0);
        assert_eq!(view.center(), PhysicalPosition::new(625.0, 500.0));
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
        assert_eq!(view.visible_image_bounds(), (0.0, 0.0, 2000.0, 1000.0));
        view.set_one_to_one(1.0);
        assert_eq!(view.visible_image_bounds(), (500.0, 0.0, 1500.0, 1000.0));
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
