use super::*;
use std::time::Instant;
use wgpu::{SurfaceColorSpace, TextureFormat};
use xl_view::color::{
    HDR_REFERENCE_WHITE_NITS, OutputEncoding, OutputPeak, OutputTransform, SourceIntensityTarget,
    composite_linear,
};
use xl_view::decode::{DecodeLimits, decode_file};

const RENDERING_OPTIONS: RenderingOptions = RenderingOptions {
    exposure_stops: 0.0,
    background: BackgroundMode::Black,
};

#[test]
fn device_prefers_memory_usage_over_allocator_throughput() {
    let descriptor = viewer_device_descriptor();

    assert!(
        descriptor
            .required_features
            .contains(wgpu::Features::CLEAR_TEXTURE)
    );
    assert!(matches!(
        descriptor.memory_hints,
        wgpu::MemoryHints::MemoryUsage
    ));
}

#[test]
fn unavailable_native_backend_has_a_distinct_category_and_guidance() {
    let error = GpuInitializationError::backend_unavailable(wgpu::RequestAdapterError::EnvNotSet);

    assert_eq!(error.category(), "gpu_backend_unavailable");
    assert!(
        error
            .to_string()
            .starts_with(super::super::backend_unavailable_message())
    );

    let output_error = GpuInitializationError::from(SurfaceOutputError::NoUsablePair);
    assert_eq!(output_error.category(), "gpu_output_unavailable");
}

#[test]
#[cfg(not(target_vendor = "apple"))]
#[ignore = "requires a Vulkan adapter for optional HDR metadata device creation"]
fn hdr_metadata_bridge_returns_a_usable_vulkan_device() {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("HDR metadata bridge test requires a Vulkan adapter");
    let descriptor = viewer_device_descriptor();
    let (device, queue, _signaler) =
        pollster::block_on(vulkan_hdr_metadata::request_device(&adapter, &descriptor)).unwrap();

    assert!(device.features().contains(wgpu::Features::CLEAR_TEXTURE));
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("HDR metadata bridge smoke-test encoder"),
    });
    queue.submit([encoder.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

#[test]
fn every_output_space_selects_its_explicit_shader_transform() {
    for (color_space, format, expected_mode, expected_encode, expected_bits) in [
        (
            SurfaceColorSpace::Srgb,
            TextureFormat::Bgra8UnormSrgb,
            0,
            false,
            8,
        ),
        (
            SurfaceColorSpace::Srgb,
            TextureFormat::Bgra8Unorm,
            0,
            true,
            8,
        ),
        (
            SurfaceColorSpace::ExtendedSrgbLinear,
            TextureFormat::Rgba16Float,
            1,
            false,
            0,
        ),
        (
            SurfaceColorSpace::Bt2100Pq,
            TextureFormat::Rgb10a2Unorm,
            2,
            false,
            10,
        ),
        (
            SurfaceColorSpace::Bt2100Hlg,
            TextureFormat::Rgb10a2Unorm,
            3,
            false,
            10,
        ),
    ] {
        let parameters = shader_parameters(
            SurfaceCandidate {
                color_space,
                format,
            },
            RENDERING_OPTIONS,
            &wgpu::DisplayHdrInfo::default(),
            None,
            SourceDynamicRange::Hlg,
            HLG_PATTERN_SOURCE_PEAK_NITS,
            (1_920, 1_080),
            None,
            true,
            false,
        );
        assert_eq!(parameters.mode, expected_mode);
        assert_eq!(parameters.encode_srgb, expected_encode);
        assert_eq!(parameters.dither_bits, expected_bits);
        assert_eq!(
            parameters.hdr_reference_white_nits.to_bits(),
            HDR_REFERENCE_WHITE_NITS.to_bits()
        );
        assert_eq!(
            parameters.source_peak_nits.to_bits(),
            HLG_PATTERN_SOURCE_PEAK_NITS.to_bits()
        );
    }
}

#[test]
fn shader_parameters_carry_exposure_as_a_uniform() {
    let options = RenderingOptions {
        exposure_stops: -1.25,
        ..RENDERING_OPTIONS
    };
    let parameters = shader_parameters(
        SurfaceCandidate {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Bgra8UnormSrgb,
        },
        options,
        &wgpu::DisplayHdrInfo::default(),
        None,
        SourceDynamicRange::Hlg,
        HLG_PATTERN_SOURCE_PEAK_NITS,
        (1_920, 1_080),
        None,
        true,
        false,
    );
    assert_eq!(parameters.exposure_stops.to_bits(), (-1.25_f32).to_bits());
}

#[test]
fn diagnostics_pattern_requires_the_explicit_startup_option() {
    let candidate = SurfaceCandidate {
        color_space: SurfaceColorSpace::Srgb,
        format: TextureFormat::Bgra8UnormSrgb,
    };
    let parameters = |image_dimensions, diagnostics_pattern, ui_centered| {
        shader_parameters(
            candidate,
            RENDERING_OPTIONS,
            &wgpu::DisplayHdrInfo::default(),
            image_dimensions,
            SourceDynamicRange::Hlg,
            HLG_PATTERN_SOURCE_PEAK_NITS,
            (1_920, 1_080),
            None,
            diagnostics_pattern,
            ui_centered,
        )
    };

    assert_eq!(parameters(None, false, true).content_mode, 0);
    assert_eq!(parameters(None, true, false).content_mode, 1);
    assert_eq!(parameters(Some((800, 600)), true, false).content_mode, 2);
    assert!(parameters(None, false, true).center_ui);
    assert_eq!(
        parameters(None, false, true).source_dynamic_range,
        SourceDynamicRange::Hlg.shader_code()
    );
}

#[test]
fn background_modes_have_stable_shader_values_and_cycle() {
    let modes = [
        (BackgroundMode::Black, 0),
        (BackgroundMode::MiddleGray, 3),
        (BackgroundMode::White, 2),
        (BackgroundMode::Checkerboard, 1),
    ];
    for (index, &(mode, shader_value)) in modes.iter().enumerate() {
        assert_eq!(background_shader_mode(mode), shader_value);
        assert_eq!(next_background(mode), modes[(index + 1) % modes.len()].0);
    }
}

#[test]
fn gpu_shaders_parse_and_validate() {
    for source in [
        presentation_shader_source(),
        include_str!("../shaders/mipmap.wgsl").to_owned(),
    ] {
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn parameter_buffer_has_uniform_and_copy_destination_usage() {
    assert!(PARAM_BUFFER_USAGE.contains(wgpu::BufferUsages::UNIFORM));
    assert!(PARAM_BUFFER_USAGE.contains(wgpu::BufferUsages::COPY_DST));
}

#[test]
fn resampling_reserves_gpu_memory_before_tiles() {
    let viewport = (3840, 2160);
    let view = ViewTransform::fit(
        PhysicalSize::new(8000, 4000),
        PhysicalSize::new(viewport.0, viewport.1),
    );
    let required =
        required_resampling_gpu_bytes(viewport, view.scale(), 16_384).expect("non-1:1 view");
    let coarse_bytes = 32 * 1024 * 1024;
    let total_bytes = coarse_bytes + required;
    let reservation =
        resampling_reservation(viewport, Some(view), 16_384, total_bytes - coarse_bytes);
    let reserved = reservation.bytes();

    assert_eq!(reservation, ResamplingReservation::Reserved(required));
    assert_eq!(reserved, required);
    assert_eq!(
        maximum_capacity(100, total_bytes - coarse_bytes - reserved, u32::MAX),
        0
    );
    assert_eq!(
        resampling_reservation(viewport, Some(view), 16_384, required - 1),
        ResamplingReservation::Insufficient {
            required,
            available: required - 1,
        }
    );

    let one_tile = rgba16f_texture_budget_bytes(
        tile_texture_extent(),
        tile_texture_extent(),
        mip_level_count(tile_texture_extent(), tile_texture_extent()),
        1,
    );
    assert_eq!(maximum_capacity(100, one_tile - 1, u32::MAX), 0);
    assert_eq!(maximum_capacity(100, one_tile, u32::MAX), 1);

    let unit_view = ViewTransform::fit(
        PhysicalSize::new(viewport.0, viewport.1),
        PhysicalSize::new(viewport.0, viewport.1),
    );
    assert_eq!(
        resampling_reservation(viewport, Some(unit_view), 16_384, u64::MAX),
        ResamplingReservation::NotRequired
    );
}

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
    let bounds = (2.1 * tile_size, 0.0, 2.9 * tile_size, tile_size - 1.0);
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
fn upload_repack_premultiplies_linear_rgb() {
    let mut encoded = [0_u8; 8];
    encode_premultiplied_rgba16f_row(&[1.0, 0.5, 0.25, 0.5], &mut encoded);
    let decoded: [f32; 4] = std::array::from_fn(|channel| {
        let start = channel * 2;
        half::f16::from_bits(u16::from_le_bytes(
            encoded[start..start + 2].try_into().unwrap(),
        ))
        .to_f32()
    });
    for (actual, expected) in decoded.into_iter().zip([0.5, 0.25, 0.125, 0.5]) {
        assert!((actual - expected).abs() < f32::EPSILON);
    }
}

#[test]
#[ignore = "requires a native GPU adapter for tile upload and mip generation"]
fn tiled_cache_uploads_one_tile_and_hits_on_repeated_view() {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = crate::gpu::native_backends();
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("tile-cache test requires a native GPU adapter");
    let mut device_descriptor = wgpu::DeviceDescriptor::default();
    device_descriptor.required_features |= wgpu::Features::CLEAR_TEXTURE;
    let (device, queue) = pollster::block_on(adapter.request_device(&device_descriptor)).unwrap();

    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/alpha.jxl");
    let image = decode_file(&fixture, DecodeLimits::from_memory_ceiling_mib(4)).unwrap();
    let store = Arc::clone(&image.store);
    let mut cache = TileCache::active(
        &device,
        &queue,
        store,
        64 * 1024 * 1024,
        (image.width, image.height),
        Arc::new(|| {}),
        Arc::new(Mutex::new(())),
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

#[test]
#[ignore = "requires a native GPU adapter for offscreen tile-boundary execution"]
fn tiled_shader_matches_continuous_sampling_across_boundaries() {
    const IMAGE_WIDTH: u32 = TILE_SIZE * 2;
    const IMAGE_HEIGHT: u32 = 2;
    const VIEWPORT_WIDTH: u32 = 8;
    const VIEWPORT_HEIGHT: u32 = 2;

    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = crate::gpu::native_backends();
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("tile-boundary test requires a native GPU adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

    let mut fixture = Vec::with_capacity(usize::try_from(IMAGE_WIDTH * IMAGE_HEIGHT * 4).unwrap());
    for _ in 0..IMAGE_HEIGHT {
        for x in 0..IMAGE_WIDTH {
            let value = f32::from(u16::try_from(x).unwrap())
                / f32::from(u16::try_from(IMAGE_WIDTH - 1).unwrap());
            fixture.extend_from_slice(&[value, 0.25, 1.0 - value, 1.0]);
        }
    }
    let cache =
        tile_cache_from_canonical_pixels(&device, &queue, &fixture, IMAGE_WIDTH, IMAGE_HEIGHT);

    for scale in [0.5, 0.75, 1.0, 1.5, 2.0] {
        let mut view = ViewTransform::fit(
            PhysicalSize::new(IMAGE_WIDTH, IMAGE_HEIGHT),
            PhysicalSize::new(VIEWPORT_WIDTH, VIEWPORT_HEIGHT),
        );
        view.set_one_to_one(scale);
        let desired_center = 511.5 + 1.5 / scale;
        view.pan_by((512.0 - desired_center) * scale, 0.0);
        let reference = render_offscreen_scene(
            &device,
            &queue,
            SurfaceColorSpace::ExtendedSrgbLinear,
            TextureFormat::Rgba16Float,
            1_000.0,
            BackgroundMode::Black,
            SourceDynamicRange::Sdr,
            Some(&fixture),
            None,
            None,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            (VIEWPORT_WIDTH, VIEWPORT_HEIGHT),
            Some(view),
        );
        let tiled = render_offscreen_scene(
            &device,
            &queue,
            SurfaceColorSpace::ExtendedSrgbLinear,
            TextureFormat::Rgba16Float,
            1_000.0,
            BackgroundMode::Black,
            SourceDynamicRange::Sdr,
            None,
            None,
            Some(&cache),
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            (VIEWPORT_WIDTH, VIEWPORT_HEIGHT),
            Some(view),
        );
        for (pixel, (reference, tiled)) in reference.iter().zip(&tiled).enumerate() {
            for (channel, (&reference, &tiled)) in reference.iter().zip(tiled).enumerate() {
                assert!(
                    (reference - tiled).abs() <= 5.0e-3,
                    "scale {scale}, pixel {pixel}, channel {channel}: reference {reference}, tiled {tiled}",
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a native GPU adapter for mip readback"]
fn linear_mip_generation_preserves_light_and_premultiplied_alpha() {
    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = crate::gpu::native_backends();
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("mip readback test requires a native GPU adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

    // Transparent white must contribute no RGB after premultiplication. The
    // 50% linear red result also distinguishes linear averaging from an
    // encoded-sRGB average decoded back to roughly 21.4% linear light.
    let pixels = [
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
    ];
    let texture = create_coarse_texture(&device, &queue, &pixels, 2, 2).unwrap();
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("linear mip readback"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 1,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
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
    let (sender, receiver) = mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = readback.get_mapped_range(..).unwrap();
    let decoded: [f32; 4] = std::array::from_fn(|channel| {
        let start = channel * 2;
        half::f16::from_bits(u16::from_le_bytes(
            mapped[start..start + 2].try_into().unwrap(),
        ))
        .to_f32()
    });
    for (actual, expected) in decoded.into_iter().zip([0.5, 0.0, 0.0, 0.5]) {
        assert!((actual - expected).abs() < f32::EPSILON);
    }
}

const OFFSCREEN_WIDTH: u32 = 7;
const OFFSCREEN_HEIGHT: u32 = 2;
const PATCH_NITS: [f32; 7] = [
    0.0,
    100.0,
    HDR_REFERENCE_WHITE_NITS,
    1_000.0,
    4_000.0,
    10_000.0,
    12_000.0,
];

struct OffscreenTestGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[derive(Clone, Copy)]
struct OffscreenRenderRequest<'a> {
    color_space: SurfaceColorSpace,
    format: TextureFormat,
    peak_nits: f32,
    background: BackgroundMode,
    source_dynamic_range: SourceDynamicRange,
    canonical_pixels: Option<&'a [f32]>,
    ui_pixel: Option<[f32; 4]>,
    tiled_resources: Option<&'a TileCache>,
}

#[derive(Clone, Copy)]
struct OffscreenOutputCase {
    color_space: SurfaceColorSpace,
    format: TextureFormat,
    encoding: OutputEncoding,
    peak_nits: f32,
    tolerance: f64,
}

impl OffscreenTestGpu {
    fn new() -> Self {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = crate::gpu::native_backends();
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .expect("offscreen color test requires a native GPU adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        Self { device, queue }
    }

    fn render(&self, request: OffscreenRenderRequest<'_>) -> Vec<[f32; 4]> {
        let has_image = request.canonical_pixels.is_some() || request.tiled_resources.is_some();
        render_offscreen_scene(
            &self.device,
            &self.queue,
            request.color_space,
            request.format,
            request.peak_nits,
            request.background,
            request.source_dynamic_range,
            request.canonical_pixels,
            request.ui_pixel,
            request.tiled_resources,
            (OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT),
            (OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT),
            has_image.then(|| {
                ViewTransform::fit(
                    PhysicalSize::new(OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT),
                    PhysicalSize::new(OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT),
                )
            }),
        )
    }
}

#[test]
#[ignore = "requires a native GPU adapter for offscreen shader execution"]
fn offscreen_shader_pixels_match_cpu_reference() {
    let gpu = OffscreenTestGpu::new();
    let luminance_fixture = canonical_luminance_fixture();

    assert_tiled_sampling_matches_canonical(&gpu);
    assert_output_transforms_match_reference(&gpu, &luminance_fixture);
    assert_sdr_attachment_paths_agree(&gpu, &luminance_fixture);
    assert_hlg_pattern_round_trips(&gpu);
    assert_alpha_compositing_matches_reference(&gpu);
    assert_transparent_pixels_use_background(&gpu);
    assert_ui_composition_matches_reference(&gpu, &luminance_fixture);
    assert_sdr_white_matches_reference(&gpu);
}

fn canonical_luminance_fixture() -> Vec<f32> {
    let mut fixture =
        Vec::with_capacity(usize::try_from(OFFSCREEN_WIDTH * OFFSCREEN_HEIGHT * 4).unwrap());
    for _ in 0..OFFSCREEN_HEIGHT {
        for nits in PATCH_NITS {
            let working = nits / HDR_REFERENCE_WHITE_NITS;
            fixture.extend_from_slice(&[working, working, working, 1.0]);
        }
    }
    fixture
}

fn tiled_color_fixture() -> Vec<f32> {
    (0..usize::try_from(OFFSCREEN_WIDTH * OFFSCREEN_HEIGHT).unwrap())
        .flat_map(|index| {
            let alpha = [0.25, 0.5, 0.75, 1.0][index % 4];
            [
                f32::from(
                    u16::try_from(index % usize::try_from(OFFSCREEN_WIDTH).unwrap()).unwrap(),
                ) / 8.0,
                f32::from(
                    u16::try_from(index / usize::try_from(OFFSCREEN_WIDTH).unwrap()).unwrap(),
                ) / 2.0,
                0.125,
                alpha,
            ]
        })
        .collect()
}

fn assert_tiled_sampling_matches_canonical(gpu: &OffscreenTestGpu) {
    let fixture = tiled_color_fixture();
    let tiled_cache = tile_cache_from_canonical_pixels(
        &gpu.device,
        &gpu.queue,
        &fixture,
        OFFSCREEN_WIDTH,
        OFFSCREEN_HEIGHT,
    );
    let reference = gpu.render(OffscreenRenderRequest {
        color_space: SurfaceColorSpace::ExtendedSrgbLinear,
        format: TextureFormat::Rgba16Float,
        peak_nits: 1_000.0,
        background: BackgroundMode::Checkerboard,
        source_dynamic_range: SourceDynamicRange::Sdr,
        canonical_pixels: Some(&fixture),
        ui_pixel: None,
        tiled_resources: None,
    });
    let tiled = gpu.render(OffscreenRenderRequest {
        color_space: SurfaceColorSpace::ExtendedSrgbLinear,
        format: TextureFormat::Rgba16Float,
        peak_nits: 1_000.0,
        background: BackgroundMode::Checkerboard,
        source_dynamic_range: SourceDynamicRange::Sdr,
        canonical_pixels: None,
        ui_pixel: None,
        tiled_resources: Some(&tiled_cache),
    });

    for (index, (reference, tiled)) in reference.iter().zip(&tiled).enumerate() {
        assert_rgba_close(
            &format!("tiled shader pixel {index}"),
            tiled,
            reference.map(f64::from),
            1.0e-3,
        );
    }
}

fn assert_output_transforms_match_reference(gpu: &OffscreenTestGpu, fixture: &[f32]) {
    let cases = [
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Bt2100Pq,
            format: TextureFormat::Rgb10a2Unorm,
            encoding: OutputEncoding::Pq,
            peak_nits: PQ_ENCODING_PEAK_NITS,
            tolerance: 2.0 / 1_023.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Bt2100Hlg,
            format: TextureFormat::Rgb10a2Unorm,
            encoding: OutputEncoding::Hlg,
            peak_nits: HLG_ENCODING_PEAK_NITS,
            tolerance: 2.0 / 1_023.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::ExtendedSrgbLinear,
            format: TextureFormat::Rgba16Float,
            encoding: OutputEncoding::ExtendedLinear,
            peak_nits: EXTENDED_LINEAR_FALLBACK_PEAK_NITS,
            tolerance: 1.5e-1,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Rgba8Unorm,
            encoding: OutputEncoding::SdrSrgbExplicit,
            peak_nits: HDR_REFERENCE_WHITE_NITS,
            tolerance: 2.0 / 255.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Rgba8UnormSrgb,
            encoding: OutputEncoding::SdrSrgbExplicit,
            peak_nits: HDR_REFERENCE_WHITE_NITS,
            tolerance: 2.0 / 255.0,
        },
    ];

    for case in cases {
        let actual = gpu.render(OffscreenRenderRequest {
            color_space: case.color_space,
            format: case.format,
            peak_nits: case.peak_nits,
            background: BackgroundMode::Black,
            source_dynamic_range: SourceDynamicRange::Hlg,
            canonical_pixels: Some(fixture),
            ui_pixel: None,
            tiled_resources: None,
        });
        let reference = output_reference(case, SourceDynamicRange::Hlg, 1_000.0);
        for (index, nits) in PATCH_NITS.map(f64::from).into_iter().enumerate() {
            let working = nits / f64::from(HDR_REFERENCE_WHITE_NITS);
            assert_rgba_close(
                &format!("{:?} patch {nits}", case.color_space),
                &actual[index],
                reference.transform([working, working, working, 1.0]),
                case.tolerance,
            );
        }
    }
}

fn assert_sdr_attachment_paths_agree(gpu: &OffscreenTestGpu, fixture: &[f32]) {
    let explicit = gpu.render(OffscreenRenderRequest {
        color_space: SurfaceColorSpace::Srgb,
        format: TextureFormat::Rgba8Unorm,
        peak_nits: HDR_REFERENCE_WHITE_NITS,
        background: BackgroundMode::Black,
        source_dynamic_range: SourceDynamicRange::Hlg,
        canonical_pixels: Some(fixture),
        ui_pixel: None,
        tiled_resources: None,
    });
    let hardware = gpu.render(OffscreenRenderRequest {
        color_space: SurfaceColorSpace::Srgb,
        format: TextureFormat::Rgba8UnormSrgb,
        peak_nits: HDR_REFERENCE_WHITE_NITS,
        background: BackgroundMode::Black,
        source_dynamic_range: SourceDynamicRange::Hlg,
        canonical_pixels: Some(fixture),
        ui_pixel: None,
        tiled_resources: None,
    });

    for (index, (explicit, hardware)) in explicit.iter().zip(&hardware).enumerate() {
        assert_rgb_close(
            &format!("sRGB attachment pixel {index}"),
            hardware,
            std::array::from_fn(|channel| f64::from(explicit[channel])),
            1.0 / 255.0,
        );
    }
    for channel in 0..3 {
        let mut previous = -1.0_f32;
        for pixel in &explicit {
            assert!((0.0..=1.0).contains(&pixel[channel]));
            assert!(pixel[channel] + 2.0 / 255.0 >= previous);
            previous = pixel[channel];
        }
    }
    assert!(
        explicit[0][..3]
            .iter()
            .all(|channel| *channel <= 1.0 / 255.0)
    );
    assert!(explicit[1][..3].iter().all(|channel| *channel > 0.5));
}

fn assert_hlg_pattern_round_trips(gpu: &OffscreenTestGpu) {
    let actual = gpu.render(OffscreenRenderRequest {
        color_space: SurfaceColorSpace::Bt2100Hlg,
        format: TextureFormat::Rgba16Float,
        peak_nits: HLG_PATTERN_SOURCE_PEAK_NITS,
        background: BackgroundMode::Black,
        source_dynamic_range: SourceDynamicRange::Hlg,
        canonical_pixels: None,
        ui_pixel: None,
        tiled_resources: None,
    });
    let grey_40 = (414.0_f32 - 64.0) / 876.0;
    let expected = [
        [grey_40, grey_40, grey_40],
        [0.75, 0.75, 0.75],
        [0.0, 0.75, 0.75],
        [0.0, 0.75, 0.0],
        [0.75, 0.0, 0.75],
        [0.0, 0.0, 0.75],
        [grey_40, grey_40, grey_40],
    ];
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_rgb_close(
            &format!("HLG source pixel {index}"),
            actual,
            expected.map(f64::from),
            1.0e-3,
        );
    }
}

fn assert_alpha_compositing_matches_reference(gpu: &OffscreenTestGpu) {
    let mut fixture =
        vec![0.0_f32; usize::try_from(OFFSCREEN_WIDTH * OFFSCREEN_HEIGHT * 4).unwrap()];
    for row in 0..usize::try_from(OFFSCREEN_HEIGHT).unwrap() {
        for (x, alpha) in [0.0, 0.5, 1.0].into_iter().enumerate() {
            let offset = (row * usize::try_from(OFFSCREEN_WIDTH).unwrap() + x) * 4;
            fixture[offset..offset + 4].copy_from_slice(&[1.0, 0.0, 0.0, alpha]);
        }
    }
    let actual = gpu.render(OffscreenRenderRequest {
        color_space: SurfaceColorSpace::ExtendedSrgbLinear,
        format: TextureFormat::Rgba16Float,
        peak_nits: 1_000.0,
        background: BackgroundMode::Checkerboard,
        source_dynamic_range: SourceDynamicRange::Hlg,
        canonical_pixels: Some(&fixture),
        ui_pixel: None,
        tiled_resources: None,
    });
    let reference = extended_linear_hlg_reference();
    for (x, alpha) in [0.0, 0.5, 1.0].into_iter().enumerate() {
        let composite = composite_linear([1.0, 0.0, 0.0, alpha], [0.18; 3]);
        assert_rgba_close(
            &format!("alpha {alpha}"),
            &actual[x],
            reference.transform([composite[0], composite[1], composite[2], 1.0]),
            5.0e-3,
        );
    }
}

fn assert_transparent_pixels_use_background(gpu: &OffscreenTestGpu) {
    let fixture = vec![1.0_f32; usize::try_from(OFFSCREEN_WIDTH * OFFSCREEN_HEIGHT * 4).unwrap()]
        .chunks_exact(4)
        .flat_map(|_| [1.0, 1.0, 1.0, 0.0])
        .collect::<Vec<_>>();
    let reference = extended_linear_hlg_reference();
    for (background, level) in [
        (BackgroundMode::Black, 0.0),
        (BackgroundMode::Checkerboard, 0.18),
        (BackgroundMode::White, 1.0),
        (BackgroundMode::MiddleGray, 0.18),
    ] {
        let actual = gpu.render(OffscreenRenderRequest {
            color_space: SurfaceColorSpace::ExtendedSrgbLinear,
            format: TextureFormat::Rgba16Float,
            peak_nits: 1_000.0,
            background,
            source_dynamic_range: SourceDynamicRange::Hlg,
            canonical_pixels: Some(&fixture),
            ui_pixel: None,
            tiled_resources: None,
        });
        assert_rgba_close(
            &format!("{background:?} background"),
            &actual[0],
            reference.transform([level, level, level, 1.0]),
            5.0e-3,
        );
    }
}

fn assert_ui_composition_matches_reference(gpu: &OffscreenTestGpu, fixture: &[f32]) {
    let cases = [
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Bt2100Pq,
            format: TextureFormat::Rgb10a2Unorm,
            encoding: OutputEncoding::Pq,
            peak_nits: PQ_ENCODING_PEAK_NITS,
            tolerance: 2.0 / 1_023.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Bt2100Hlg,
            format: TextureFormat::Rgb10a2Unorm,
            encoding: OutputEncoding::Hlg,
            peak_nits: HLG_ENCODING_PEAK_NITS,
            tolerance: 2.0 / 1_023.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::ExtendedSrgbLinear,
            format: TextureFormat::Rgba16Float,
            encoding: OutputEncoding::ExtendedLinear,
            peak_nits: EXTENDED_LINEAR_FALLBACK_PEAK_NITS,
            tolerance: 5.0e-3,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Rgba8UnormSrgb,
            encoding: OutputEncoding::SdrSrgbExplicit,
            peak_nits: HDR_REFERENCE_WHITE_NITS,
            tolerance: 1.0 / 255.0,
        },
    ];

    for case in cases {
        let actual = gpu.render(OffscreenRenderRequest {
            color_space: case.color_space,
            format: case.format,
            peak_nits: case.peak_nits,
            background: BackgroundMode::Black,
            source_dynamic_range: SourceDynamicRange::Hlg,
            canonical_pixels: Some(fixture),
            ui_pixel: Some([1.0; 4]),
            tiled_resources: None,
        });
        let reference = output_reference(case, SourceDynamicRange::Hlg, 1_000.0);
        let expected = if case.color_space == SurfaceColorSpace::Srgb {
            [1.0; 4]
        } else {
            reference.transform([1.0; 4])
        };
        assert_rgba_close(
            &format!("{:?} UI pixel", case.color_space),
            &actual[0],
            expected,
            case.tolerance,
        );
    }
}

fn assert_sdr_white_matches_reference(gpu: &OffscreenTestGpu) {
    let fixture = vec![1.0_f32; usize::try_from(OFFSCREEN_WIDTH * OFFSCREEN_HEIGHT * 4).unwrap()];
    let cases = [
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Bt2100Pq,
            format: TextureFormat::Rgb10a2Unorm,
            encoding: OutputEncoding::Pq,
            peak_nits: PQ_ENCODING_PEAK_NITS,
            tolerance: 2.0 / 1_023.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Bt2100Hlg,
            format: TextureFormat::Rgb10a2Unorm,
            encoding: OutputEncoding::Hlg,
            peak_nits: 1_000.0,
            tolerance: 2.0 / 1_023.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::ExtendedSrgbLinear,
            format: TextureFormat::Rgba16Float,
            encoding: OutputEncoding::ExtendedLinear,
            peak_nits: 1_000.0,
            tolerance: 5.0e-3,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Rgba8UnormSrgb,
            encoding: OutputEncoding::SdrSrgbExplicit,
            peak_nits: HDR_REFERENCE_WHITE_NITS,
            tolerance: 1.0 / 255.0,
        },
    ];

    for case in cases {
        let actual = gpu.render(OffscreenRenderRequest {
            color_space: case.color_space,
            format: case.format,
            peak_nits: case.peak_nits,
            background: BackgroundMode::Black,
            source_dynamic_range: SourceDynamicRange::Sdr,
            canonical_pixels: Some(&fixture),
            ui_pixel: None,
            tiled_resources: None,
        });
        let reference = output_reference(case, SourceDynamicRange::Sdr, HDR_REFERENCE_WHITE_NITS);
        assert_rgba_close(
            &format!("{:?} SDR white", case.color_space),
            &actual[0],
            reference.transform([1.0; 4]),
            case.tolerance,
        );
    }
}

fn output_reference(
    case: OffscreenOutputCase,
    source_dynamic_range: SourceDynamicRange,
    source_peak_nits: f32,
) -> OutputTransform {
    OutputTransform {
        encoding: case.encoding,
        source_dynamic_range,
        source_intensity_target: SourceIntensityTarget::new(f64::from(source_peak_nits)).unwrap(),
        output_peak: OutputPeak::new(f64::from(case.peak_nits)).unwrap(),
        exposure_stops: 0.0,
    }
}

fn extended_linear_hlg_reference() -> OutputTransform {
    OutputTransform {
        encoding: OutputEncoding::ExtendedLinear,
        source_dynamic_range: SourceDynamicRange::Hlg,
        source_intensity_target: SourceIntensityTarget::new(1_000.0).unwrap(),
        output_peak: OutputPeak::new(1_000.0).unwrap(),
        exposure_stops: 0.0,
    }
}

fn assert_rgba_close(context: &str, actual: &[f32; 4], expected: [f64; 4], tolerance: f64) {
    for (channel, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (f64::from(actual) - expected).abs() <= tolerance,
            "{context} channel {channel}: expected {expected}, got {actual}",
        );
    }
}

fn assert_rgb_close(context: &str, actual: &[f32; 4], expected: [f64; 3], tolerance: f64) {
    for (channel, (&actual, expected)) in actual[..3].iter().zip(expected).enumerate() {
        assert!(
            (f64::from(actual) - expected).abs() <= tolerance,
            "{context} channel {channel}: expected {expected}, got {actual}",
        );
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // The offscreen harness keeps every shader binding and readback step visible to its callers.
fn render_offscreen_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_space: SurfaceColorSpace,
    format: TextureFormat,
    peak_nits: f32,
    background: BackgroundMode,
    source_dynamic_range: SourceDynamicRange,
    canonical_pixels: Option<&[f32]>,
    ui_pixel: Option<[f32; 4]>,
    tiled_resources: Option<&TileCache>,
    image_dimensions: (u32, u32),
    viewport_dimensions: (u32, u32),
    view_transform: Option<ViewTransform>,
) -> Vec<[f32; 4]> {
    const BYTES_PER_ROW: u32 = 256;
    let (image_width, image_height) = image_dimensions;
    let (viewport_width, viewport_height) = viewport_dimensions;
    let pixel_bytes = match format {
        TextureFormat::Rgba16Float => 8,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgb10a2Unorm => {
            4
        }
        _ => unreachable!("offscreen test used unsupported target format"),
    };
    assert!(viewport_width * pixel_bytes <= BYTES_PER_ROW);
    let candidate = SurfaceCandidate {
        color_space,
        format,
    };
    let rendering_options = RenderingOptions {
        exposure_stops: 0.0,
        background,
    };
    let hdr_info = wgpu::DisplayHdrInfo {
        luminance: (color_space == SurfaceColorSpace::Srgb).then_some(wgpu::DisplayLuminance {
            sdr_white_nits: Some(peak_nits),
            ..Default::default()
        }),
        ..Default::default()
    };
    let coarse_texture = canonical_pixels
        .map_or_else(
            || create_coarse_texture(device, queue, &[0.0; 4], 1, 1),
            |pixels| create_coarse_texture(device, queue, pixels, image_width, image_height),
        )
        .unwrap();
    let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    let ui_pixel = ui_pixel.unwrap_or([0.0; 4]);
    let ui_texture = create_ui_texture(device, queue, &ui_pixel, 1, 1).unwrap();
    assert_eq!(ui_texture.mip_level_count(), 1);
    let has_image = canonical_pixels.is_some() || tiled_resources.is_some();
    let mut params = shader_parameters(
        candidate,
        rendering_options,
        &hdr_info,
        has_image.then_some(image_dimensions),
        source_dynamic_range,
        HLG_PATTERN_SOURCE_PEAK_NITS,
        viewport_dimensions,
        view_transform,
        true,
        false,
    );
    let fallback_tile_cache;
    let selected_tile_cache = if let Some(tile_cache) = tiled_resources {
        params.tiled_image = true;
        params.tile_columns = u32::try_from(tile_cache.mapping.len()).unwrap();
        tile_cache
    } else {
        fallback_tile_cache = TileCache::fallback(device);
        &fallback_tile_cache
    };
    let presentation = PresentationResources::new(
        device,
        candidate.format,
        params,
        &coarse_texture,
        selected_tile_cache,
        &coarse_texture,
        &ui_texture,
        &image_sampler,
    );
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen color reference target"),
        size: wgpu::Extent3d {
            width: viewport_width,
            height: viewport_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: candidate.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen color reference readback"),
        size: u64::from(BYTES_PER_ROW * viewport_height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("offscreen color reference pass"),
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
        pass.set_pipeline(&presentation.pipeline);
        pass.set_bind_group(0, &presentation.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(BYTES_PER_ROW),
                rows_per_image: Some(viewport_height),
            },
        },
        wgpu::Extent3d {
            width: viewport_width,
            height: viewport_height,
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
    let pixels = decode_readback_row(&mapped, candidate.format, viewport_width);
    drop(mapped);
    readback.unmap();
    pixels
}

#[allow(clippy::cast_possible_truncation)] // Fixture dimensions and packed half-float channels are deliberately small.
fn tile_cache_from_canonical_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
) -> TileCache {
    assert!(height <= TILE_SIZE);
    assert_eq!(pixels.len(), usize::try_from(width * height * 4).unwrap());

    let extent = usize::try_from(tile_texture_extent()).unwrap();
    let width = usize::try_from(width).unwrap();
    let height = usize::try_from(height).unwrap();
    let gutter = usize::try_from(TILE_GUTTER).unwrap();
    let tile_size = usize::try_from(TILE_SIZE).unwrap();
    let columns = width.div_ceil(tile_size);
    let mut source_row = vec![0.0_f32; extent * 4];
    let mut encoded_pixels = vec![0_u8; extent * extent * 8];
    let mapping = (0..u32::try_from(columns).unwrap()).collect::<Vec<_>>();
    let cache = TileCache {
        texture: create_tile_array_texture(device, u32::try_from(columns).unwrap()),
        mapping_buffer: create_tile_mapping_buffer(device, &mapping),
        mapping: mapping.clone(),
        slot_tiles: (0..u32::try_from(columns).unwrap()).map(Some).collect(),
        last_used: vec![0; columns],
        epoch: 0,
        hits: 0,
        misses: 0,
        tile_columns: u32::try_from(columns).unwrap(),
        tile_rows: 1,
        coarse_downsample: 1,
        desired_tiles: Vec::new(),
        request_generation: 0,
        last_view_center: None,
        worker: None,
    };
    for column in 0..columns {
        let origin_x = column * tile_size;
        for tile_y in 0..extent {
            let source_y = tile_y.saturating_sub(gutter).min(height - 1);
            for tile_x in 0..extent {
                let source_x = origin_x
                    .saturating_add(tile_x)
                    .saturating_sub(gutter)
                    .min(width - 1);
                let source_offset = (source_y * width + source_x) * 4;
                source_row[tile_x * 4..tile_x * 4 + 4]
                    .copy_from_slice(&pixels[source_offset..source_offset + 4]);
            }
            encode_premultiplied_rgba16f_row(
                &source_row,
                &mut encoded_pixels[tile_y * extent * 8..(tile_y + 1) * extent * 8],
            );
        }
        upload_tile_layer(
            queue,
            &cache.texture,
            u32::try_from(column).unwrap(),
            &encoded_pixels,
        )
        .unwrap();
    }
    let mut mip_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offscreen tiled fixture mip encoder"),
    });
    generate_linear_mips_for_layers(
        device,
        &mut mip_encoder,
        &cache.texture,
        mip_level_count(tile_texture_extent(), tile_texture_extent()),
        &mapping,
    );
    queue.submit([mip_encoder.finish()]);
    cache
}

#[allow(clippy::cast_precision_loss)] // Packed values are at most ten bits.
fn decode_readback_row(bytes: &[u8], format: TextureFormat, width: u32) -> Vec<[f32; 4]> {
    let width = width as usize;
    match format {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => bytes[..width * 4]
            .chunks_exact(4)
            .map(|pixel| {
                [pixel[0], pixel[1], pixel[2], pixel[3]].map(|channel| f32::from(channel) / 255.0)
            })
            .collect(),
        TextureFormat::Rgb10a2Unorm => bytes[..width * 4]
            .chunks_exact(4)
            .map(|pixel| {
                let packed = u32::from_le_bytes(pixel.try_into().unwrap());
                [
                    (packed & 0x3ff) as f32 / 1_023.0,
                    ((packed >> 10) & 0x3ff) as f32 / 1_023.0,
                    ((packed >> 20) & 0x3ff) as f32 / 1_023.0,
                    ((packed >> 30) & 0x3) as f32 / 3.0,
                ]
            })
            .collect(),
        TextureFormat::Rgba16Float => bytes[..width * 8]
            .chunks_exact(8)
            .map(|pixel| {
                std::array::from_fn(|channel| {
                    let start = channel * 2;
                    half::f16::from_bits(u16::from_le_bytes(
                        pixel[start..start + 2].try_into().unwrap(),
                    ))
                    .to_f32()
                })
            })
            .collect(),
        _ => unreachable!("offscreen test used unsupported readback format"),
    }
}

#[test]
fn gpu_failures_have_actionable_messages() {
    let device_lost = GpuFailure::DeviceLost {
        reason: wgpu::DeviceLostReason::Unknown,
        message: "driver reset".to_owned(),
    };
    assert_eq!(
        device_lost.to_string(),
        "the GPU device was lost (Unknown): driver reset; restart xl-view"
    );
    assert_eq!(device_lost.category(), "gpu_device_lost");
    let out_of_memory = GpuFailure::OutOfMemory;
    assert!(out_of_memory.to_string().contains("smaller image"));
    assert_eq!(out_of_memory.category(), "gpu_out_of_memory");
    assert_eq!(
        GpuFailure::Validation("bad binding".to_owned()).to_string(),
        "the GPU rejected a rendering operation: bad binding"
    );
    assert_eq!(
        GpuFailure::Internal("queue failure".to_owned()).to_string(),
        "an internal GPU error occurred: queue failure"
    );
}
