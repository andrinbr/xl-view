use super::*;

fn distinct_parameters() -> PresentationParams {
    PresentationParams {
        pq_output: true,
        content_mode: ContentMode::DecodedImage,
        background_mode: 17,
        hdr_reference_white_nits: 5.0,
        source_peak_nits: 6.0,
        output_peak_nits: 7.0,
        exposure_stops: 8.0,
        effective_source_peak_pq: 9.0,
        output_peak_pq: 10.0,
        viewport_width: 11,
        viewport_height: 12,
        image_width: 13,
        image_height: 14,
        view_center_x: 15.0,
        view_center_y: 16.0,
        view_scale: 17.0,
        center_ui: true,
        tiled_image: true,
        tile_size: 22,
        tile_columns: 23,
        tile_gutter: 24,
        resampled_viewport: true,
    }
}

#[test]
fn parameters_serialize_in_wgsl_word_order() {
    assert_eq!(
        distinct_parameters().to_words(),
        [
            1,                  // 0: pq_output
            2,                  // 1: content_mode
            17,                 // 2: background_mode
            5.0_f32.to_bits(),  // 3: hdr_reference_white_nits
            6.0_f32.to_bits(),  // 4: source_peak_nits
            7.0_f32.to_bits(),  // 5: output_peak_nits
            8.0_f32.to_bits(),  // 6: exposure_stops
            9.0_f32.to_bits(),  // 7: effective_source_peak_pq
            10.0_f32.to_bits(), // 8: output_peak_pq
            11,                 // 9: viewport_width
            12,                 // 10: viewport_height
            13,                 // 11: image_width
            14,                 // 12: image_height
            15.0_f32.to_bits(), // 13: view_center_x
            16.0_f32.to_bits(), // 14: view_center_y
            17.0_f32.to_bits(), // 15: view_scale
            1,                  // 16: center_ui
            1,                  // 17: tiled_image
            22,                 // 18: tile_size
            23,                 // 19: tile_columns
            24,                 // 20: tile_gutter
            1,                  // 21: resampled_viewport
        ]
    );
}

#[test]
fn content_modes_match_the_shader_constants() {
    let shader = include_str!("../shaders/presentation.wgsl");
    for (mode, shader_name) in [
        (ContentMode::EmptyState, "CONTENT_EMPTY_STATE"),
        (
            ContentMode::DiagnosticsPattern,
            "CONTENT_DIAGNOSTICS_PATTERN",
        ),
        (ContentMode::DecodedImage, "CONTENT_DECODED_IMAGE"),
    ] {
        let expected = format!("const {shader_name}: u32 = {}u;", mode as u32);
        assert!(
            shader.contains(&expected),
            "shader must contain `{expected}`"
        );
    }
}

#[test]
fn rust_serializer_tracks_the_wgsl_field_sequence() {
    let shader = include_str!("../shaders/presentation.wgsl");
    let (_, after_start) = shader
        .split_once("struct Params {")
        .expect("presentation shader declares Params");
    let (body, _) = after_start
        .split_once('}')
        .expect("presentation Params declaration is closed");
    let fields = body
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_suffix(',')
                .and_then(|line| line.split_once(':'))
                .map(|(name, _)| name.trim())
        })
        .collect::<Vec<_>>();
    assert_eq!(
        fields,
        [
            "pq_output",
            "content_mode",
            "background_mode",
            "hdr_reference_white_nits",
            "source_peak_nits",
            "output_peak_nits",
            "exposure_stops",
            "effective_source_peak_pq",
            "output_peak_pq",
            "viewport_width",
            "viewport_height",
            "image_width",
            "image_height",
            "view_center_x",
            "view_center_y",
            "view_scale",
            "center_ui",
            "tiled_image",
            "tile_size",
            "tile_columns",
            "tile_gutter",
            "resampled_viewport",
        ]
    );
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

// Evaluates `dither_encoded` for every probed position, bit depth, and level.
const DITHER_PROBE_SHADER: &str = r"
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8)
fn probe_dither(@builtin(global_invocation_id) id: vec3u) {
    let position = id.xy * 251u;
    var index = (id.y * 16u + id.x) * 30u;
    for (var depth = 0u; depth < 2u; depth++) {
        for (var level = 0u; level < 5u; level++) {
            let dithered = dither_encoded(vec3f(f32(level) * 0.25), position, 8u + depth * 2u);
            output[index] = dithered.x;
            output[index + 1u] = dithered.y;
            output[index + 2u] = dithered.z;
            index += 3u;
        }
    }
}
";

#[test]
#[ignore = "requires a native GPU adapter for compute shader execution"]
fn shader_dither_matches_cpu_reference() {
    const GRID: u32 = 16;
    const SAMPLES_PER_POSITION: u32 = 30;
    let (device, queue) = crate::gpu::test_utils::request_native_test_gpu(
        crate::gpu::native_backends(),
        wgpu::Features::empty(),
        "dither parity test requires a native GPU adapter",
    );
    let source = [include_str!("../shaders/dither.wgsl"), DITHER_PROBE_SHADER].join("\n");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dither parity probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dither parity probe"),
        layout: None,
        module: &module,
        entry_point: Some("probe_dither"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let size = u64::from(GRID * GRID * SAMPLES_PER_POSITION) * 4;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dither parity output"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dither parity readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dither parity bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(GRID / 8, GRID / 8, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = readback.get_mapped_range(..).unwrap();
    let samples = mapped
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    drop(mapped);
    readback.unmap();

    let mut samples = samples.into_iter();
    for y in 0..GRID {
        for x in 0..GRID {
            for bits in [8_u8, 10] {
                for level in 0..5_u8 {
                    let value = f64::from(level) * 0.25;
                    for channel in 0..3 {
                        let expected =
                            xl_view::color::dither_unorm(value, x * 251, y * 251, channel, bits);
                        let actual = f64::from(samples.next().unwrap());
                        // f32 hashing on the GPU differs from the f64 reference by about 1e-7.
                        assert!(
                            (actual - expected).abs() <= 1.0e-6,
                            "position ({x}, {y}), {bits} bits, value {value}, channel {channel}: \
                             shader {actual}, CPU {expected}",
                        );
                    }
                }
            }
        }
    }
}
