use wgpu::TextureFormat;
use wgpu::util::DeviceExt;

use super::tiles::TileCache;

pub(super) const PARAM_BUFFER_USAGE: wgpu::BufferUsages =
    wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST);
const PRESENTATION_PARAM_WORD_COUNT: usize = 28;

/// Named CPU representation of the WGSL `Params` uniform.
///
/// [`Self::to_words`] is the single source of truth for the cross-language
/// field order and inserts the padding required by the shader layout.
#[allow(clippy::struct_excessive_bools)] // These are four independent WGSL flags serialized as u32 words, not overlapping state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PresentationParams {
    pub(super) mode: u32,
    pub(super) encode_srgb: bool,
    pub(super) dither_bits: u32,
    pub(super) content_mode: u32,
    pub(super) hdr_reference_white_nits: f32,
    pub(super) source_peak_nits: f32,
    pub(super) output_peak_nits: f32,
    pub(super) exposure_stops: f32,
    pub(super) viewport_width: u32,
    pub(super) viewport_height: u32,
    pub(super) image_width: u32,
    pub(super) image_height: u32,
    pub(super) ui_white_nits: f32,
    pub(super) view_center_x: f32,
    pub(super) view_center_y: f32,
    pub(super) view_scale: f32,
    pub(super) background_mode: u32,
    pub(super) center_ui: bool,
    pub(super) source_dynamic_range: u32,
    pub(super) tiled_image: bool,
    pub(super) tile_size: u32,
    pub(super) tile_columns: u32,
    pub(super) tile_gutter: u32,
    pub(super) resampled_viewport: bool,
}

impl PresentationParams {
    fn to_words(self) -> [u32; PRESENTATION_PARAM_WORD_COUNT] {
        [
            self.mode,
            u32::from(self.encode_srgb),
            self.dither_bits,
            self.content_mode,
            self.hdr_reference_white_nits.to_bits(),
            self.source_peak_nits.to_bits(),
            self.output_peak_nits.to_bits(),
            self.exposure_stops.to_bits(),
            self.viewport_width,
            self.viewport_height,
            self.image_width,
            self.image_height,
            self.ui_white_nits.to_bits(),
            self.view_center_x.to_bits(),
            self.view_center_y.to_bits(),
            self.view_scale.to_bits(),
            self.background_mode,
            u32::from(self.center_ui),
            self.source_dynamic_range,
            0,
            u32::from(self.tiled_image),
            self.tile_size,
            self.tile_columns,
            self.tile_gutter,
            u32::from(self.resampled_viewport),
            0,
            0,
            0,
        ]
    }

    pub(super) fn to_ne_bytes(self) -> [[u8; 4]; PRESENTATION_PARAM_WORD_COUNT] {
        self.to_words().map(u32::to_ne_bytes)
    }
}

pub(super) struct PresentationResources {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bind_group: wgpu::BindGroup,
    pub(super) params_buffer: wgpu::Buffer,
}

impl PresentationResources {
    #[allow(clippy::too_many_arguments)] // The binding inputs remain explicit at this small call surface.
    pub(super) fn new(
        device: &wgpu::Device,
        target_format: TextureFormat,
        params: PresentationParams,
        coarse_texture: &wgpu::Texture,
        tile_cache: &TileCache,
        resampled_texture: &wgpu::Texture,
        ui_texture: &wgpu::Texture,
        image_sampler: &wgpu::Sampler,
    ) -> Self {
        let pipeline = create_presentation_pipeline(device, target_format);
        let params = params.to_ne_bytes();
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("presentation parameters"),
            contents: params.as_flattened(),
            usage: PARAM_BUFFER_USAGE,
        });
        let bind_group = create_presentation_bind_group(
            device,
            &pipeline,
            &params_buffer,
            coarse_texture,
            tile_cache,
            resampled_texture,
            ui_texture,
            image_sampler,
        );
        Self {
            pipeline,
            bind_group,
            params_buffer,
        }
    }

    pub(super) fn rebuild_bind_group(
        &mut self,
        device: &wgpu::Device,
        coarse_texture: &wgpu::Texture,
        tile_cache: &TileCache,
        resampled_texture: &wgpu::Texture,
        ui_texture: &wgpu::Texture,
        image_sampler: &wgpu::Sampler,
    ) {
        self.bind_group = create_presentation_bind_group(
            device,
            &self.pipeline,
            &self.params_buffer,
            coarse_texture,
            tile_cache,
            resampled_texture,
            ui_texture,
            image_sampler,
        );
    }

    #[allow(clippy::too_many_arguments)] // Mirrors construction when the surface format changes.
    pub(super) fn rebuild_pipeline(
        &mut self,
        device: &wgpu::Device,
        target_format: TextureFormat,
        coarse_texture: &wgpu::Texture,
        tile_cache: &TileCache,
        resampled_texture: &wgpu::Texture,
        ui_texture: &wgpu::Texture,
        image_sampler: &wgpu::Sampler,
    ) {
        self.pipeline = create_presentation_pipeline(device, target_format);
        self.rebuild_bind_group(
            device,
            coarse_texture,
            tile_cache,
            resampled_texture,
            ui_texture,
            image_sampler,
        );
    }
}

pub(super) fn presentation_shader_source() -> String {
    [
        include_str!("shaders/color_transform.wgsl"),
        include_str!("shaders/test_pattern.wgsl"),
        include_str!("shaders/dither.wgsl"),
        include_str!("shaders/presentation.wgsl"),
    ]
    .join("\n")
}

fn create_presentation_pipeline(
    device: &wgpu::Device,
    target_format: TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("presentation shader"),
        source: wgpu::ShaderSource::Wgsl(presentation_shader_source().into()),
    });
    super::create_fullscreen_render_pipeline(
        device,
        "presentation pipeline",
        &shader,
        target_format,
    )
}

#[allow(clippy::too_many_arguments)] // Each entry maps directly to one shader resource binding.
fn create_presentation_bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    params_buffer: &wgpu::Buffer,
    coarse_texture: &wgpu::Texture,
    tile_cache: &TileCache,
    resampled_texture: &wgpu::Texture,
    ui_texture: &wgpu::Texture,
    image_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let coarse_view = coarse_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let ui_view = ui_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let resampled_view = resampled_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let tile_view = tile_cache
        .texture
        .create_view(&wgpu::TextureViewDescriptor {
            label: Some("canonical high-resolution tile-cache view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("presentation bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&coarse_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(image_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&ui_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&tile_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: tile_cache.mapping_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&resampled_view),
            },
        ],
    })
}

#[cfg(test)]
mod tests;
