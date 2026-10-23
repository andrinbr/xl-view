use std::io;

use wgpu::TextureFormat;

use super::upload::TextureUploadLayout;
use xl_view::units::{BYTES_PER_MIB, usize_from_u32};

const GPU_TEXTURE_OVERHEAD_DIVISOR: u64 = 10;

pub(super) fn create_rgba16f_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
    mip_level_count: u32,
    label: &'static str,
) -> Result<wgpu::Texture, io::Error> {
    let upload_layout = TextureUploadLayout::rgba16f(width, height, label)?;
    let expected_samples = usize_from_u32(width)
        .checked_mul(usize_from_u32(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| io::Error::other(format!("{label} sample count overflowed")))?;
    if pixels.len() != expected_samples {
        return Err(io::Error::other(format!(
            "{label} expected {expected_samples} samples, got {}",
            pixels.len()
        )));
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let row_samples = usize_from_u32(width) * 4;
    let mut staging = upload_layout.allocate_staging();
    for stripe in upload_layout.stripes() {
        for stripe_row in 0..stripe.row_count() {
            let source_row = stripe.first_row() + stripe_row;
            let source_start = usize_from_u32(source_row) * row_samples;
            let source = &pixels[source_start..source_start + row_samples];
            let destination = upload_layout.staging_row_mut(&mut staging, stripe_row);
            encode_premultiplied_rgba16f_row(source, destination);
        }
        upload_layout.write_stripe(queue, &texture, 0, &staging, stripe);
    }
    if mip_level_count > 1 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("linear RGBA16F mip encoder"),
        });
        generate_linear_mips_for_layers(device, &mut encoder, &texture, mip_level_count, &[0]);
        queue.submit([encoder.finish()]);
    }
    Ok(texture)
}

pub(super) fn encode_premultiplied_rgba16f_row(source: &[f32], destination: &mut [u8]) {
    for (rgba, encoded) in source.chunks_exact(4).zip(destination.chunks_exact_mut(8)) {
        let alpha = if rgba[3].is_finite() {
            rgba[3].clamp(0.0, 1.0)
        } else {
            0.0
        };
        for channel in 0..3 {
            let bytes = half::f16::from_f32(rgba[channel] * alpha)
                .to_bits()
                .to_le_bytes();
            encoded[channel * 2..channel * 2 + 2].copy_from_slice(&bytes);
        }
        encoded[6..8].copy_from_slice(&half::f16::from_f32(alpha).to_bits().to_le_bytes());
    }
}

pub(super) fn mip_level_count(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).leading_zeros()
}

/// Conservative budget charge for an RGBA16F texture allocation.
///
/// wgpu does not expose the driver's allocation size. Account for 256-byte
/// row alignment, add ten percent for implementation overhead, and round to a
/// whole MiB.
pub(super) fn rgba16f_texture_budget_bytes(
    width: u32,
    height: u32,
    mip_levels: u32,
    array_layers: u32,
) -> u64 {
    if width == 0 || height == 0 || mip_levels == 0 || array_layers == 0 {
        return 0;
    }
    let alignment = u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let mut total = 0_u64;
    let (mut width, mut height) = (width, height);
    for _ in 0..mip_levels {
        let row_bytes = u64::from(width).saturating_mul(8);
        let padded_row_bytes = row_bytes
            .checked_next_multiple_of(alignment)
            .unwrap_or(u64::MAX);
        total = total.saturating_add(
            padded_row_bytes
                .saturating_mul(u64::from(height))
                .saturating_mul(u64::from(array_layers)),
        );
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    let with_overhead = total.saturating_add(total.div_ceil(GPU_TEXTURE_OVERHEAD_DIVISOR));
    with_overhead
        .checked_next_multiple_of(BYTES_PER_MIB)
        .unwrap_or(u64::MAX)
}

pub(super) fn generate_linear_mips_for_layers(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    mip_level_count: u32,
    layers: &[u32],
) {
    if mip_level_count <= 1 {
        return;
    }
    LinearMipGenerator::new(device).generate(device, encoder, texture, mip_level_count, layers);
}

pub(super) struct LinearMipGenerator {
    sampler: wgpu::Sampler,
    pipeline: wgpu::RenderPipeline,
}

impl LinearMipGenerator {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/mipmap.wgsl"));
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear-light mip generator sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let pipeline = super::create_fullscreen_render_pipeline(
            device,
            "linear-light mip generator pipeline",
            &shader,
            TextureFormat::Rgba16Float,
        );
        Self { sampler, pipeline }
    }

    pub(super) fn generate(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        mip_level_count: u32,
        layers: &[u32],
    ) {
        if mip_level_count <= 1 {
            return;
        }
        let layout = self.pipeline.get_bind_group_layout(0);
        for &layer in layers {
            for destination_level in 1..mip_level_count {
                let source_view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("linear-light mip source"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: destination_level - 1,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let destination_view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("linear-light mip destination"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: destination_level,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("linear-light mip generator bind group"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&source_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("linear-light mip generation pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &destination_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_utils::request_native_test_gpu;

    #[test]
    fn mip_level_count_covers_the_longest_dimension() {
        assert_eq!(mip_level_count(1, 1), 1);
        assert_eq!(mip_level_count(3, 5), 3);
    }

    #[test]
    fn texture_budget_includes_alignment_margin_and_mib_rounding() {
        let logical = 3840_u64 * 2160 * 8;
        let budget = rgba16f_texture_budget_bytes(3840, 2160, 1, 1);

        assert!(budget > logical);
        assert_eq!(budget % BYTES_PER_MIB, 0);
        assert_eq!(budget, 70 * BYTES_PER_MIB);
    }

    #[test]
    fn texture_budget_accounts_for_mips_and_array_layers() {
        let one_layer = rgba16f_texture_budget_bytes(528, 528, mip_level_count(528, 528), 1);
        let four_layers = rgba16f_texture_budget_bytes(528, 528, mip_level_count(528, 528), 4);

        assert!(four_layers > one_layer);
        assert!(four_layers <= one_layer.saturating_mul(4));
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
    #[ignore = "requires a native GPU adapter for mip readback"]
    fn linear_mip_generation_preserves_light_and_premultiplied_alpha() {
        let (device, queue) = request_native_test_gpu(
            crate::gpu::native_backends(),
            wgpu::Features::empty(),
            "mip readback test requires a native GPU adapter",
        );

        // Transparent white must contribute no RGB after premultiplication. The
        // 50% linear red result also distinguishes linear averaging from an
        // encoded-sRGB average decoded back to roughly 21.4% linear light.
        let pixels = [
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
        ];
        let texture = create_rgba16f_texture(
            &device,
            &queue,
            &pixels,
            2,
            2,
            mip_level_count(2, 2),
            "linear mip test",
        )
        .unwrap();
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
        let (sender, receiver) = std::sync::mpsc::channel();
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
}
