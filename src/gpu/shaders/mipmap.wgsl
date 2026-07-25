@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    return VertexOutput(vec4f(x, y, 0.0, 1.0), vec2f(x, -y) * 0.5 + 0.5);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    // The source is premultiplied, display-linear BT.2020. Filtering therefore
    // averages light and coverage without encoded-space or alpha-edge artifacts.
    return textureSampleLevel(source_texture, source_sampler, input.uv, 0.0);
}
