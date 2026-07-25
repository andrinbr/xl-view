use std::sync::OnceLock;

use resvg::{tiny_skia, usvg};
use winit::icon::{Icon, RgbaIcon};

const ICON_SVG_BYTES: &[u8] = include_bytes!("../assets/icons/xl-view.svg");
const WINDOW_ICON_SIZE: u32 = 128;

pub(crate) fn window_icon() -> Icon {
    RgbaIcon::new(
        rasterize(WINDOW_ICON_SIZE),
        WINDOW_ICON_SIZE,
        WINDOW_ICON_SIZE,
    )
    .expect("the rasterized application icon dimensions must match its pixels")
    .into()
}

#[allow(clippy::cast_precision_loss)] // Production callers request practical window and UI icon sizes.
pub(crate) fn rasterize(size: u32) -> Vec<u8> {
    let tree = icon_tree();
    let mut pixmap =
        tiny_skia::Pixmap::new(size, size).expect("the application icon must fit in memory");
    let source_size = tree.size();
    let transform = tiny_skia::Transform::from_scale(
        size as f32 / source_size.width(),
        size as f32 / source_size.height(),
    );
    resvg::render(tree, transform, &mut pixmap.as_mut());

    pixmap.take_demultiplied()
}

fn icon_tree() -> &'static usvg::Tree {
    static TREE: OnceLock<usvg::Tree> = OnceLock::new();
    TREE.get_or_init(|| {
        usvg::Tree::from_data(ICON_SVG_BYTES, &usvg::Options::default())
            .expect("the packaged xl-view icon must remain valid SVG")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rasterized_icon_has_straight_rgba_pixels() {
        let pixels = rasterize(64);

        assert_eq!(pixels.len(), 64 * 64 * 4);
        assert!(pixels.chunks_exact(4).any(|pixel| pixel[3] == 0));
        assert!(pixels.chunks_exact(4).any(|pixel| pixel[3] == 255));
        assert!(
            pixels
                .chunks_exact(4)
                .any(|pixel| pixel[3] > 0 && pixel[3] < 255)
        );
    }
}
