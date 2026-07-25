fn hash_to_unit(x: u32, y: u32, channel: u32, phase: u32) -> f32 {
    var value = x * 0x9e3779b9u + y * 0x85ebca6bu + channel * 0xc2b2ae35u + phase * 0x27d4eb2du;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return f32(value) / 4294967295.0;
}

fn dither_noise(position: vec2u, channel: u32, bits: u32) -> f32 {
    if bits == 0u { return 0.0; }
    let levels = f32((1u << bits) - 1u);
    return (hash_to_unit(position.x, position.y, channel, 0u)
        - hash_to_unit(position.x, position.y, channel, 1u)) / levels;
}

fn dither_encoded(encoded: vec3f, position: vec2u, bits: u32) -> vec3f {
    return clamp(encoded + vec3f(
        dither_noise(position, 0u, bits),
        dither_noise(position, 1u, bits),
        dither_noise(position, 2u, bits),
    ), vec3f(0.0), vec3f(1.0));
}
