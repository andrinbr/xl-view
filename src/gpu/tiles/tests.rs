use std::time::Instant;

use winit::dpi::PhysicalSize;
use xl_view::decode::{DecodeLimits, decode_file};

use super::*;
use crate::gpu::test_utils::request_native_test_gpu;
use crate::test_utils::fixture_path;

#[test]
fn tile_working_set_covers_a_one_to_one_viewport_with_a_border() {
    assert_eq!(interactive_working_set_tiles((3840, 1600)), 77);
    assert_eq!(interactive_working_set_tiles((1024, 682)), 25);
    assert_eq!(interactive_working_set_tiles((0, 682)), 0);

    assert_eq!(
        working_set_capacity(247, (3840, 1600), u64::MAX, u32::MAX),
        77
    );
    assert_eq!(
        working_set_capacity(12, (3840, 1600), u64::MAX, u32::MAX),
        12
    );
}

#[test]
fn visible_tiles_stay_center_first_and_directional_prefetch_stays_bounded() {
    let tile_size = f64::from(TILE_SIZE);
    let bounds = ImageBounds {
        left: 2.1 * tile_size,
        top: 0.0,
        right: 2.9 * tile_size,
        bottom: tile_size - 1.0,
    };
    let (stationary, center) = prioritized_tiles(bounds, 6, 1, None, 6);
    assert_eq!(stationary, [2, 1, 3]);
    assert!(same_tile_set(&stationary, &[3, 2, 1]));

    let previous_center = Some((center.0 - tile_size, center.1));
    let (moving_right, _) = prioritized_tiles(bounds, 6, 1, previous_center, 6);
    assert_eq!(moving_right, [2, 3, 4, 1]);

    let (budgeted, _) = prioritized_tiles(bounds, 6, 1, previous_center, 2);
    assert_eq!(budgeted, [2, 3]);
}

#[test]
fn newer_tile_generation_discards_queued_stale_work() {
    let old_jobs = [
        TileJob {
            generation: 7,
            logical_tile: 4,
            slot: 0,
        },
        TileJob {
            generation: 7,
            logical_tile: 5,
            slot: 1,
        },
    ];
    let mut state = TileWorkerState {
        generation: 7,
        pending: old_jobs.into(),
        shutdown: false,
    };

    assert_eq!(replace_pending_jobs(&mut state, 8), old_jobs);
    assert_eq!(state.generation, 8);
    assert!(state.pending.is_empty());
}

#[test]
#[ignore = "requires a native GPU adapter for tile upload and mip generation"]
fn tiled_cache_uploads_one_tile_and_hits_on_repeated_view() {
    let (device, queue) = request_native_test_gpu(
        crate::gpu::native_backends(),
        wgpu::Features::CLEAR_TEXTURE,
        "tile-cache test requires a native GPU adapter",
    );

    let fixture = fixture_path("alpha.jxl");
    let image = decode_file(&fixture, DecodeLimits::from_memory_ceiling_mib(4)).unwrap();
    let store = Arc::clone(&image.store);
    let workers = WorkerContext {
        device: device.clone(),
        queue: queue.clone(),
        work_ready: Arc::new(|| {}),
        submission_lock: Arc::new(Mutex::new(())),
    };
    let mut cache = TileCache::active(
        &workers,
        store,
        64 * 1024 * 1024,
        (image.width, image.height),
    )
    .unwrap();
    let initialized =
        read_rgba16f_texture_pixel(&device, &queue, &cache.texture, 3, wgpu::Origin3d::ZERO);
    assert!(initialized.iter().all(|channel| channel.to_bits() == 0));
    let mut view = ViewTransform::fit(
        PhysicalSize::new(image.width, image.height),
        PhysicalSize::new(image.width, image.height),
    );
    view.zoom_at(
        winit::dpi::PhysicalPosition::new(
            f64::from(image.width) / 2.0,
            f64::from(image.height) / 2.0,
        ),
        2.0,
    );
    cache.request_view(&queue, view);
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    while cache.mapping[0] == MISSING_TILE_SLOT {
        cache.process_completions(&queue).unwrap();
        assert!(
            Instant::now() < deadline,
            "asynchronous tile upload did not complete"
        );
        std::thread::yield_now();
    }
    assert_eq!(cache.mapping, [0]);
    assert_eq!(cache.misses, 1);
    cache.request_view(&queue, view);
    assert_eq!(cache.hits, 1);

    let pixel = read_rgba16f_texture_pixel(
        &device,
        &queue,
        &cache.texture,
        0,
        wgpu::Origin3d {
            x: TILE_GUTTER,
            y: TILE_GUTTER,
            z: 0,
        },
    );
    assert!((pixel[3] - 1.0).abs() < f32::EPSILON);
}

fn read_rgba16f_texture_pixel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    origin: wgpu::Origin3d,
) -> [f32; 4] {
    const BYTES_PER_ROW: u32 = 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile-cache test readback"),
        size: u64::from(BYTES_PER_ROW),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(BYTES_PER_ROW),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = readback.get_mapped_range(..).unwrap();
    let pixel = std::array::from_fn(|channel| {
        let offset = channel * 2;
        half::f16::from_bits(u16::from_le_bytes(
            mapped[offset..offset + 2].try_into().unwrap(),
        ))
        .to_f32()
    });
    drop(mapped);
    readback.unmap();
    pixel
}
