use super::*;
use crate::gpu::test_utils::request_native_test_gpu;
use crate::test_utils::fixture_path;
use wgpu::{SurfaceColorSpace, TextureFormat};
use xl_view::color::{
    HDR_REFERENCE_WHITE_NITS, OutputEncoding, OutputPeak, OutputTransform, SourceIntensityTarget,
    composite_linear, srgb_eotf,
};
use xl_view::decode::{DecodeLimits, decode_file};

const RENDERING_OPTIONS: RenderingOptions = RenderingOptions {
    exposure_stops: 0.0,
    background: BackgroundMode::Black,
};

fn test_surface_selection(candidate: SurfaceCandidate, peak_nits: f32) -> SurfaceSelection {
    let display_target = Ok(peak_nits);
    let capabilities = [wgpu::SurfaceFormatCapabilities {
        format: candidate.format,
        color_spaces: candidate
            .color_space
            .to_color_spaces()
            .expect("tests use an explicit color space"),
    }];
    select_required_surface(
        &capabilities,
        if candidate.color_space == SurfaceColorSpace::Bt2100Pq {
            OutputMode::Pq
        } else {
            OutputMode::Sdr
        },
        display_target,
    )
    .unwrap()
}

#[cfg(target_os = "linux")]
#[test]
fn forced_pq_bootstraps_until_wayland_identifies_the_window_output() {
    let capabilities = [
        wgpu::SurfaceFormatCapabilities {
            format: TextureFormat::Rgb10a2Unorm,
            color_spaces: wgpu::SurfaceColorSpaces::BT2100_PQ,
        },
        wgpu::SurfaceFormatCapabilities {
            format: TextureFormat::Bgra8UnormSrgb,
            color_spaces: wgpu::SurfaceColorSpaces::SRGB,
        },
    ];
    let unmatched = Err(HdrUnavailableReason::OutputUnmatched);

    let bootstrap =
        select_surface_with_wayland_bootstrap(&capabilities, OutputMode::Pq, unmatched, None)
            .unwrap();
    assert_eq!(bootstrap.candidate().color_space, SurfaceColorSpace::Srgb);
    assert_eq!(bootstrap.display_target(), unmatched);

    assert_eq!(
        select_surface_with_wayland_bootstrap(
            &capabilities,
            OutputMode::Pq,
            unmatched,
            Some("DP-1"),
        ),
        Err(SurfaceOutputError::HdrTargetUnavailable(
            HdrUnavailableReason::OutputUnmatched
        ))
    );
    assert_eq!(
        select_surface_with_wayland_bootstrap(
            &capabilities,
            OutputMode::Pq,
            Err(HdrUnavailableReason::MissingDisplayInformation),
            None,
        ),
        Err(SurfaceOutputError::HdrTargetUnavailable(
            HdrUnavailableReason::MissingDisplayInformation
        ))
    );
}

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
fn hdr_metadata_describes_only_the_mapped_pq_output_peak() {
    let pq_selection = test_surface_selection(
        SurfaceCandidate {
            color_space: SurfaceColorSpace::Bt2100Pq,
            format: TextureFormat::Rgb10a2Unorm,
        },
        750.0,
    );
    let sdr_selection = test_surface_selection(
        SurfaceCandidate {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Bgra8UnormSrgb,
        },
        750.0,
    );

    assert_eq!(
        output_hdr_metadata(pq_selection),
        Some(HdrMetadata::display_mapped_bt2020(750.0))
    );
    assert_eq!(output_hdr_metadata(sdr_selection), None);
}

#[test]
fn unavailable_native_backend_has_a_distinct_category_and_guidance() {
    let backend = super::super::native_backend_order()[0];
    let error = GpuInitializationError::backend_unavailable(
        backend.unavailable_message(),
        wgpu::RequestAdapterError::EnvNotSet,
    );

    assert_eq!(error.category(), "gpu_backend_unavailable");
    assert!(error.to_string().starts_with(backend.unavailable_message()));

    let output_error = GpuInitializationError::from(SurfaceOutputError::SrgbSurfaceUnavailable);
    assert_eq!(output_error.category(), "gpu_output_unavailable");
}

#[test]
fn final_output_spaces_select_their_explicit_shader_transform() {
    for (color_space, format, expected_pq_output, peak_nits) in [
        (
            SurfaceColorSpace::Srgb,
            TextureFormat::Bgra8UnormSrgb,
            false,
            HDR_REFERENCE_WHITE_NITS,
        ),
        (
            SurfaceColorSpace::Bt2100Pq,
            TextureFormat::Rgb10a2Unorm,
            true,
            600.0,
        ),
    ] {
        let parameters = shader_parameters(
            test_surface_selection(
                SurfaceCandidate {
                    color_space,
                    format,
                },
                peak_nits,
            ),
            RENDERING_OPTIONS,
            None,
            HLG_PATTERN_SOURCE_PEAK_NITS,
            (1_920, 1_080),
            None,
            true,
            false,
        );
        assert_eq!(parameters.pq_output, expected_pq_output);
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
        test_surface_selection(
            SurfaceCandidate {
                color_space: SurfaceColorSpace::Srgb,
                format: TextureFormat::Bgra8UnormSrgb,
            },
            HDR_REFERENCE_WHITE_NITS,
        ),
        options,
        None,
        HLG_PATTERN_SOURCE_PEAK_NITS,
        (1_920, 1_080),
        None,
        true,
        false,
    );
    assert_eq!(parameters.exposure_stops.to_bits(), (-1.25_f32).to_bits());
    let effective_source_peak =
        HDR_REFERENCE_WHITE_NITS.max(HLG_PATTERN_SOURCE_PEAK_NITS * (-1.25_f32).exp2());
    assert_eq!(
        parameters.effective_source_peak_pq.to_bits(),
        pq_oetf(effective_source_peak / 10_000.0).to_bits(),
    );
    assert_eq!(
        parameters.output_peak_pq.to_bits(),
        pq_oetf(HDR_REFERENCE_WHITE_NITS / 10_000.0).to_bits(),
    );
}

#[test]
fn diagnostics_pattern_requires_the_explicit_startup_option() {
    let candidate = SurfaceCandidate {
        color_space: SurfaceColorSpace::Srgb,
        format: TextureFormat::Bgra8UnormSrgb,
    };
    let parameters = |image_dimensions, diagnostics_pattern, ui_centered| {
        shader_parameters(
            test_surface_selection(candidate, HDR_REFERENCE_WHITE_NITS),
            RENDERING_OPTIONS,
            image_dimensions,
            HLG_PATTERN_SOURCE_PEAK_NITS,
            (1_920, 1_080),
            None,
            diagnostics_pattern,
            ui_centered,
        )
    };

    assert_eq!(
        parameters(None, false, true).content_mode,
        ContentMode::EmptyState
    );
    assert_eq!(
        parameters(None, true, false).content_mode,
        ContentMode::DiagnosticsPattern
    );
    assert_eq!(
        parameters(Some((800, 600)), true, false).content_mode,
        ContentMode::DecodedImage
    );
    assert!(parameters(None, false, true).center_ui);
}

#[test]
fn unloaded_source_peak_matches_the_visible_content() {
    assert_eq!(
        unloaded_source_peak_nits(false).to_bits(),
        HDR_REFERENCE_WHITE_NITS.to_bits()
    );
    assert_eq!(
        unloaded_source_peak_nits(true).to_bits(),
        HLG_PATTERN_SOURCE_PEAK_NITS.to_bits()
    );
}

#[test]
fn background_modes_have_stable_shader_values_and_cycle() {
    let shader = include_str!("../shaders/presentation.wgsl");
    let modes = [
        (BackgroundMode::Black, "BACKGROUND_BLACK"),
        (BackgroundMode::MiddleGray, "BACKGROUND_MIDDLE_GRAY"),
        (BackgroundMode::White, "BACKGROUND_WHITE"),
        (BackgroundMode::Checkerboard, "BACKGROUND_CHECKERBOARD"),
    ];
    for (index, &(mode, shader_name)) in modes.iter().enumerate() {
        let expected = format!(
            "const {shader_name}: u32 = {}u;",
            background_shader_mode(mode)
        );
        assert!(
            shader.contains(&expected),
            "shader must contain `{expected}`"
        );
        assert_eq!(next_background(mode), modes[(index + 1) % modes.len()].0);
    }
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
#[ignore = "requires a native GPU adapter for offscreen tile-boundary execution"]
fn tiled_shader_matches_continuous_sampling_across_boundaries() {
    const IMAGE_WIDTH: u32 = TILE_SIZE * 2;
    const IMAGE_HEIGHT: u32 = 2;
    const VIEWPORT_WIDTH: u32 = 8;
    const VIEWPORT_HEIGHT: u32 = 2;

    let (device, queue) = request_native_test_gpu(
        crate::gpu::native_backends(),
        wgpu::Features::empty(),
        "tile-boundary test requires a native GPU adapter",
    );

    let mut fixture = Vec::with_capacity(usize::try_from(IMAGE_WIDTH * IMAGE_HEIGHT * 4).unwrap());
    for _ in 0..IMAGE_HEIGHT {
        for x in 0..IMAGE_WIDTH {
            let value = f32::from(u16::try_from(x).unwrap())
                / f32::from(u16::try_from(IMAGE_WIDTH - 1).unwrap());
            fixture.extend_from_slice(&[value, 0.25, 1.0 - value, 1.0]);
        }
    }
    let cache =
        TileCache::from_canonical_pixels(&device, &queue, &fixture, IMAGE_WIDTH, IMAGE_HEIGHT);

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
            SurfaceColorSpace::Srgb,
            TextureFormat::Rgba8UnormSrgb,
            HDR_REFERENCE_WHITE_NITS,
            BackgroundMode::Black,
            0.0,
            HDR_REFERENCE_WHITE_NITS,
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
            SurfaceColorSpace::Srgb,
            TextureFormat::Rgba8UnormSrgb,
            HDR_REFERENCE_WHITE_NITS,
            BackgroundMode::Black,
            0.0,
            HDR_REFERENCE_WHITE_NITS,
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
    source_peak_nits: f32,
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
        // Production image state clears tile textures, as the real device allows.
        let (device, queue) = request_native_test_gpu(
            crate::gpu::native_backends(),
            wgpu::Features::CLEAR_TEXTURE,
            "offscreen color test requires a native GPU adapter",
        );
        Self { device, queue }
    }

    fn render(&self, request: OffscreenRenderRequest<'_>) -> Vec<[f32; 4]> {
        self.render_with_exposure(request, 0.0)
    }

    fn render_with_exposure(
        &self,
        request: OffscreenRenderRequest<'_>,
        exposure_stops: f32,
    ) -> Vec<[f32; 4]> {
        let has_image = request.canonical_pixels.is_some() || request.tiled_resources.is_some();
        render_offscreen_scene(
            &self.device,
            &self.queue,
            request.color_space,
            request.format,
            request.peak_nits,
            request.background,
            exposure_stops,
            request.source_peak_nits,
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
    assert_exposure_matches_reference(&gpu, &luminance_fixture);
    assert_hdr_color_mapping_matches_reference(&gpu);
    assert_alpha_compositing_matches_reference(&gpu);
    assert_transparent_pixels_use_background(&gpu);
    assert_ui_composition_matches_reference(&gpu, &luminance_fixture);
    assert_sdr_white_matches_reference(&gpu);
    assert_sdr_chromatic_fallback_matches_reference(&gpu);
    assert_negative_pixels_match_reference(&gpu);
    assert_measured_linear_peak_reaches_presentation(&gpu);
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
    let tiled_cache = TileCache::from_canonical_pixels(
        &gpu.device,
        &gpu.queue,
        &fixture,
        OFFSCREEN_WIDTH,
        OFFSCREEN_HEIGHT,
    );
    let reference = gpu.render(OffscreenRenderRequest {
        color_space: SurfaceColorSpace::Srgb,
        format: TextureFormat::Rgba8UnormSrgb,
        peak_nits: HDR_REFERENCE_WHITE_NITS,
        background: BackgroundMode::Checkerboard,
        source_peak_nits: HDR_REFERENCE_WHITE_NITS,
        canonical_pixels: Some(&fixture),
        ui_pixel: None,
        tiled_resources: None,
    });
    let tiled = gpu.render(OffscreenRenderRequest {
        color_space: SurfaceColorSpace::Srgb,
        format: TextureFormat::Rgba8UnormSrgb,
        peak_nits: HDR_REFERENCE_WHITE_NITS,
        background: BackgroundMode::Checkerboard,
        source_peak_nits: HDR_REFERENCE_WHITE_NITS,
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
            peak_nits: 600.0,
            tolerance: 2.0 / 1_023.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Rgba8UnormSrgb,
            encoding: OutputEncoding::SdrSrgbHardware,
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
            source_peak_nits: HLG_PATTERN_SOURCE_PEAK_NITS,
            canonical_pixels: Some(fixture),
            ui_pixel: None,
            tiled_resources: None,
        });
        let reference = output_reference(case, 1_000.0);
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

fn assert_exposure_matches_reference(gpu: &OffscreenTestGpu, fixture: &[f32]) {
    let cases = [
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Bt2100Pq,
            format: TextureFormat::Rgb10a2Unorm,
            encoding: OutputEncoding::Pq,
            peak_nits: HDR_REFERENCE_WHITE_NITS,
            tolerance: 2.0 / 1_023.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Rgba8UnormSrgb,
            encoding: OutputEncoding::SdrSrgbHardware,
            peak_nits: HDR_REFERENCE_WHITE_NITS,
            tolerance: 2.0 / 255.0,
        },
    ];
    for case in cases {
        for (source_peak_nits, exposure_stops) in [(1_000.0, -1.0), (HDR_REFERENCE_WHITE_NITS, 1.0)]
        {
            let actual = gpu.render_with_exposure(
                OffscreenRenderRequest {
                    color_space: case.color_space,
                    format: case.format,
                    peak_nits: case.peak_nits,
                    background: BackgroundMode::Black,
                    source_peak_nits,
                    canonical_pixels: Some(fixture),
                    ui_pixel: None,
                    tiled_resources: None,
                },
                exposure_stops,
            );
            let reference = OutputTransform {
                exposure_stops: f64::from(exposure_stops),
                ..output_reference(case, source_peak_nits)
            };
            for (index, nits) in PATCH_NITS.map(f64::from).into_iter().enumerate() {
                let working = nits / f64::from(HDR_REFERENCE_WHITE_NITS);
                assert_rgba_close(
                    &format!(
                        "{:?} source peak {source_peak_nits} exposure {exposure_stops:+.1}, patch {nits}",
                        case.color_space
                    ),
                    &actual[index],
                    reference.transform([working, working, working, 1.0]),
                    case.tolerance,
                );
            }
        }
    }
}

fn assert_hdr_color_mapping_matches_reference(gpu: &OffscreenTestGpu) {
    let colors_nits = [
        [1_000.0, 250.0, 40.0],
        [50.0, 600.0, 120.0],
        [40.0, 200.0, 1_000.0],
        [400.0, 400.0, 400.0],
        [203.0, 203.0, 203.0],
        [20.0, 5.0, 1.0],
        [0.0, 0.0, 0.0],
    ];
    let fixture = (0..OFFSCREEN_HEIGHT)
        .flat_map(|_| {
            colors_nits.into_iter().flat_map(|rgb| {
                [
                    rgb[0] / HDR_REFERENCE_WHITE_NITS,
                    rgb[1] / HDR_REFERENCE_WHITE_NITS,
                    rgb[2] / HDR_REFERENCE_WHITE_NITS,
                    1.0,
                ]
            })
        })
        .collect::<Vec<_>>();
    for case in [
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Bt2100Pq,
            format: TextureFormat::Rgb10a2Unorm,
            encoding: OutputEncoding::Pq,
            peak_nits: 600.0,
            tolerance: 2.0 / 1_023.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Rgba8UnormSrgb,
            encoding: OutputEncoding::SdrSrgbHardware,
            peak_nits: HDR_REFERENCE_WHITE_NITS,
            tolerance: 2.0 / 255.0,
        },
    ] {
        let actual = gpu.render(OffscreenRenderRequest {
            color_space: case.color_space,
            format: case.format,
            peak_nits: case.peak_nits,
            background: BackgroundMode::Black,
            source_peak_nits: HLG_PATTERN_SOURCE_PEAK_NITS,
            canonical_pixels: Some(&fixture),
            ui_pixel: None,
            tiled_resources: None,
        });
        let reference = output_reference(case, 1_000.0);
        for (index, rgb) in colors_nits.into_iter().enumerate() {
            let working = rgb.map(|channel| f64::from(channel / HDR_REFERENCE_WHITE_NITS));
            assert_rgba_close(
                &format!("{:?} HDR color {index}", case.color_space),
                &actual[index],
                reference.transform([working[0], working[1], working[2], 1.0]),
                case.tolerance,
            );
        }
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
        color_space: SurfaceColorSpace::Srgb,
        format: TextureFormat::Rgba8UnormSrgb,
        peak_nits: HDR_REFERENCE_WHITE_NITS,
        background: BackgroundMode::Checkerboard,
        source_peak_nits: HLG_PATTERN_SOURCE_PEAK_NITS,
        canonical_pixels: Some(&fixture),
        ui_pixel: None,
        tiled_resources: None,
    });
    let reference = output_reference(
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Rgba8UnormSrgb,
            encoding: OutputEncoding::SdrSrgbHardware,
            peak_nits: HDR_REFERENCE_WHITE_NITS,
            tolerance: 2.0 / 255.0,
        },
        1_000.0,
    );
    for (x, alpha) in [0.0, 0.5, 1.0].into_iter().enumerate() {
        let composite = composite_linear([1.0, 0.0, 0.0, alpha], [0.18; 3]);
        assert_rgba_close(
            &format!("alpha {alpha}"),
            &actual[x],
            reference.transform([composite[0], composite[1], composite[2], 1.0]),
            2.0 / 255.0,
        );
    }
}

fn assert_transparent_pixels_use_background(gpu: &OffscreenTestGpu) {
    let fixture = vec![1.0_f32; usize::try_from(OFFSCREEN_WIDTH * OFFSCREEN_HEIGHT * 4).unwrap()]
        .chunks_exact(4)
        .flat_map(|_| [1.0, 1.0, 1.0, 0.0])
        .collect::<Vec<_>>();
    let case = OffscreenOutputCase {
        color_space: SurfaceColorSpace::Srgb,
        format: TextureFormat::Rgba8UnormSrgb,
        encoding: OutputEncoding::SdrSrgbHardware,
        peak_nits: HDR_REFERENCE_WHITE_NITS,
        tolerance: 2.0 / 255.0,
    };
    let reference = output_reference(case, 1_000.0);
    for (background, level) in [
        (BackgroundMode::Black, 0.0),
        (BackgroundMode::Checkerboard, 0.18),
        (BackgroundMode::White, 1.0),
        (BackgroundMode::MiddleGray, 0.18),
    ] {
        let actual = gpu.render(OffscreenRenderRequest {
            color_space: case.color_space,
            format: case.format,
            peak_nits: case.peak_nits,
            background,
            source_peak_nits: HLG_PATTERN_SOURCE_PEAK_NITS,
            canonical_pixels: Some(&fixture),
            ui_pixel: None,
            tiled_resources: None,
        });
        assert_rgba_close(
            &format!("{background:?} background"),
            &actual[0],
            reference.transform([level, level, level, 1.0]),
            case.tolerance,
        );
    }
}

fn assert_ui_composition_matches_reference(gpu: &OffscreenTestGpu, fixture: &[f32]) {
    let cases = [
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Bt2100Pq,
            format: TextureFormat::Rgb10a2Unorm,
            encoding: OutputEncoding::Pq,
            peak_nits: 600.0,
            tolerance: 2.0 / 1_023.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Rgba8UnormSrgb,
            encoding: OutputEncoding::SdrSrgbHardware,
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
            source_peak_nits: HLG_PATTERN_SOURCE_PEAK_NITS,
            canonical_pixels: Some(fixture),
            ui_pixel: Some([1.0; 4]),
            tiled_resources: None,
        });
        let expected = output_reference(case, HDR_REFERENCE_WHITE_NITS).transform([1.0; 4]);
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
            peak_nits: 600.0,
            tolerance: 2.0 / 1_023.0,
        },
        OffscreenOutputCase {
            color_space: SurfaceColorSpace::Srgb,
            format: TextureFormat::Rgba8UnormSrgb,
            encoding: OutputEncoding::SdrSrgbHardware,
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
            source_peak_nits: HDR_REFERENCE_WHITE_NITS,
            canonical_pixels: Some(&fixture),
            ui_pixel: None,
            tiled_resources: None,
        });
        let reference = output_reference(case, HDR_REFERENCE_WHITE_NITS);
        assert_rgba_close(
            &format!("{:?} SDR white", case.color_space),
            &actual[0],
            reference.transform([1.0; 4]),
            case.tolerance,
        );
    }
}

fn assert_sdr_chromatic_fallback_matches_reference(gpu: &OffscreenTestGpu) {
    let working = [
        500.0 / HDR_REFERENCE_WHITE_NITS,
        0.0,
        800.0 / HDR_REFERENCE_WHITE_NITS,
        1.0,
    ];
    let fixture = working
        .into_iter()
        .cycle()
        .take(usize::try_from(OFFSCREEN_WIDTH * OFFSCREEN_HEIGHT * 4).unwrap())
        .collect::<Vec<_>>();
    let case = OffscreenOutputCase {
        color_space: SurfaceColorSpace::Srgb,
        format: TextureFormat::Rgba8UnormSrgb,
        encoding: OutputEncoding::SdrSrgbHardware,
        peak_nits: HDR_REFERENCE_WHITE_NITS,
        tolerance: 1.0 / 255.0,
    };
    let actual = gpu.render(OffscreenRenderRequest {
        color_space: case.color_space,
        format: case.format,
        peak_nits: case.peak_nits,
        background: BackgroundMode::Black,
        source_peak_nits: HDR_REFERENCE_WHITE_NITS,
        canonical_pixels: Some(&fixture),
        ui_pixel: None,
        tiled_resources: None,
    });
    let reference = output_reference(case, HDR_REFERENCE_WHITE_NITS);
    assert_rgba_close(
        "SDR chromatic fallback",
        &actual[0],
        reference.transform(working.map(f64::from)),
        case.tolerance,
    );
}

fn assert_negative_pixels_match_reference(gpu: &OffscreenTestGpu) {
    let patches = [
        [-0.25, 0.5, 0.5, 0.5],
        [-0.25, 0.5, 0.5, 1.0],
        [0.5, -0.25, 0.5, 1.0],
        [0.5, 0.5, -0.25, 1.0],
        [0.25, 0.5, 0.5, 1.0],
        [-0.25, 0.5, 0.5, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ];
    let fixture: Vec<f32> = patches.iter().flatten().copied().cycle().take(56).collect();
    for (color_space, format, encoding, tolerance) in [
        (
            SurfaceColorSpace::Srgb,
            TextureFormat::Rgba8UnormSrgb,
            OutputEncoding::SdrSrgbHardware,
            2.0 / 255.0,
        ),
        (
            SurfaceColorSpace::Bt2100Pq,
            TextureFormat::Rgb10a2Unorm,
            OutputEncoding::Pq,
            2.0 / 1023.0,
        ),
    ] {
        let case = OffscreenOutputCase {
            color_space,
            format,
            encoding,
            peak_nits: HDR_REFERENCE_WHITE_NITS,
            tolerance,
        };
        let actual = gpu.render(OffscreenRenderRequest {
            color_space,
            format,
            peak_nits: HDR_REFERENCE_WHITE_NITS,
            background: BackgroundMode::White,
            source_peak_nits: HDR_REFERENCE_WHITE_NITS,
            canonical_pixels: Some(&fixture),
            ui_pixel: None,
            tiled_resources: None,
        });
        let reference = output_reference(case, HDR_REFERENCE_WHITE_NITS);
        for (index, patch) in patches.into_iter().enumerate() {
            let composite = composite_linear(patch.map(f64::from), [1.0; 3]);
            assert_rgba_close(
                "negative image RGB",
                &actual[index],
                reference.transform([composite[0], composite[1], composite[2], 1.0]),
                tolerance,
            );
        }
    }
}

fn assert_measured_linear_peak_reaches_presentation(gpu: &OffscreenTestGpu) {
    const MEASURED_PEAK_NITS: f32 = 812.0;
    let image = decode_file(
        &fixture_path("linear-highlights.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    let viewport = (OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);
    assert_eq!((image.width, image.height), viewport);
    let workers = WorkerContext {
        device: gpu.device.clone(),
        queue: gpu.queue.clone(),
        work_ready: Arc::new(|| {}),
        submission_lock: Arc::new(Mutex::new(())),
    };
    let (state, _) = create_image_state(
        &workers,
        viewport,
        64 * 1024 * 1024,
        &image,
        ViewTransform::fit(
            PhysicalSize::new(image.width, image.height),
            PhysicalSize::new(viewport.0, viewport.1),
        ),
    )
    .unwrap();
    assert_eq!(
        state.source_peak_nits.to_bits(),
        MEASURED_PEAK_NITS.to_bits(),
        "image state peak {} nits",
        state.source_peak_nits
    );

    let case = OffscreenOutputCase {
        color_space: SurfaceColorSpace::Srgb,
        format: TextureFormat::Rgba8UnormSrgb,
        encoding: OutputEncoding::SdrSrgbHardware,
        peak_nits: HDR_REFERENCE_WHITE_NITS,
        tolerance: 2.0 / 255.0,
    };
    let patches: Vec<[f64; 4]> = image
        .store
        .coarse_pixels()
        .chunks_exact(4)
        .map(|pixel| std::array::from_fn(|c| f64::from(pixel[c])))
        .collect();
    let reference = output_reference(case, MEASURED_PEAK_NITS);
    // The 255-nit metadata, 1,000-nit fallback and SDR white each move some
    // patch by more than twice the tolerance, so routing one would fail.
    for wrong_peak in [255.0, 1_000.0, HDR_REFERENCE_WHITE_NITS] {
        let wrong = output_reference(case, wrong_peak);
        assert!(patches.iter().any(|&patch| {
            let (expected, other) = (reference.transform(patch), wrong.transform(patch));
            (0..3).any(|c| (expected[c] - other[c]).abs() > 2.0 * case.tolerance)
        }));
    }
    let actual = gpu.render(OffscreenRenderRequest {
        color_space: case.color_space,
        format: case.format,
        peak_nits: case.peak_nits,
        background: BackgroundMode::Black,
        source_peak_nits: state.source_peak_nits,
        canonical_pixels: Some(image.store.coarse_pixels()),
        ui_pixel: None,
        tiled_resources: None,
    });
    for (pixel, patch) in actual.iter().zip(patches) {
        assert_rgba_close(
            "measured linear peak",
            pixel,
            reference.transform(patch),
            case.tolerance,
        );
    }
}

fn output_reference(case: OffscreenOutputCase, source_peak_nits: f32) -> OutputTransform {
    OutputTransform {
        encoding: case.encoding,
        source_intensity_target: SourceIntensityTarget::new(f64::from(source_peak_nits)).unwrap(),
        output_peak: OutputPeak::new(f64::from(case.peak_nits)).unwrap(),
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

#[expect(clippy::too_many_arguments, clippy::too_many_lines)] // The offscreen harness keeps every shader binding and readback step visible to its callers.
fn render_offscreen_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_space: SurfaceColorSpace,
    format: TextureFormat,
    peak_nits: f32,
    background: BackgroundMode,
    exposure_stops: f32,
    source_peak_nits: f32,
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
        TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgb10a2Unorm => 4,
        _ => unreachable!("offscreen test used unsupported target format"),
    };
    assert!(viewport_width * pixel_bytes <= BYTES_PER_ROW);
    let candidate = SurfaceCandidate {
        color_space,
        format,
    };
    let rendering_options = RenderingOptions {
        exposure_stops,
        background,
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
        test_surface_selection(candidate, peak_nits),
        rendering_options,
        has_image.then_some(image_dimensions),
        source_peak_nits,
        viewport_dimensions,
        view_transform,
        true,
        false,
    );
    let fallback_tile_cache;
    let selected_tile_cache = if let Some(tile_cache) = tiled_resources {
        params.tiled_image = true;
        params.tile_columns = tile_cache.tile_columns();
        tile_cache
    } else {
        fallback_tile_cache = TileCache::fallback(device);
        &fallback_tile_cache
    };
    let presentation = PresentationResources::new(
        device,
        candidate.format,
        params,
        &PresentationBindings {
            coarse_texture: &coarse_texture,
            tile_cache: selected_tile_cache,
            resampled_texture: &coarse_texture,
            ui_texture: &ui_texture,
            image_sampler: &image_sampler,
        },
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

#[expect(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Packed values are at most ten bits; decoded sRGB fits f32.
fn decode_readback_row(bytes: &[u8], format: TextureFormat, width: u32) -> Vec<[f32; 4]> {
    let width = width as usize;
    match format {
        TextureFormat::Rgba8UnormSrgb => bytes[..width * 4]
            .chunks_exact(4)
            .map(|pixel| {
                let encoded =
                    [pixel[0], pixel[1], pixel[2]].map(|channel| f64::from(channel) / 255.0);
                [
                    srgb_eotf(encoded[0]) as f32,
                    srgb_eotf(encoded[1]) as f32,
                    srgb_eotf(encoded[2]) as f32,
                    f32::from(pixel[3]) / 255.0,
                ]
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
