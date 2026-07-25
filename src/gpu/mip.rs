use std::io;

use wgpu::TextureFormat;

use super::upload::TextureUploadLayout;
use crate::units::usize_from_u32;

const MIB_BYTES: u64 = 1024 * 1024;
const GPU_TEXTURE_OVERHEAD_DIVISOR: u64 = 10;

#[allow(clippy::too_many_arguments)] // Texture dimensions, mip policy, and diagnostic label are explicit upload inputs.
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
    let expected_samples = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
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
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: stripe.first_row(),
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            upload_layout.stripe_data(&staging, stripe),
            upload_layout.copy_buffer_layout(stripe),
            upload_layout.copy_extent(stripe),
        );
    }
    if mip_level_count > 1 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("linear RGBA16F mip encoder"),
        });
        generate_linear_mips(device, &mut encoder, &texture, mip_level_count);
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
            .saturating_add(alignment - 1)
            .checked_div(alignment)
            .unwrap_or(u64::MAX)
            .saturating_mul(alignment);
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
        .checked_add(MIB_BYTES - 1)
        .map_or(u64::MAX, |bytes| bytes / MIB_BYTES * MIB_BYTES)
}

fn generate_linear_mips(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    mip_level_count: u32,
) {
    generate_linear_mips_for_layers(device, encoder, texture, mip_level_count, &[0]);
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

    #[allow(clippy::too_many_lines)] // One loop owns the per-level views, bindings, render passes, and layer transitions.
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
        assert_eq!(budget % MIB_BYTES, 0);
        assert_eq!(budget, 70 * MIB_BYTES);
    }

    #[test]
    fn texture_budget_accounts_for_mips_and_array_layers() {
        let one_layer = rgba16f_texture_budget_bytes(528, 528, mip_level_count(528, 528), 1);
        let four_layers = rgba16f_texture_budget_bytes(528, 528, mip_level_count(528, 528), 4);

        assert!(four_layers > one_layer);
        assert!(four_layers <= one_layer.saturating_mul(4));
    }
}
