//! Shared support for GPU unit tests, which exist only in the binary crate.

pub(super) fn request_native_test_gpu(
    backends: wgpu::Backends,
    required_features: wgpu::Features,
    requirement: &str,
) -> (wgpu::Device, wgpu::Queue) {
    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = backends;
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("{requirement}: {error}"));
    let mut device_descriptor = wgpu::DeviceDescriptor::default();
    device_descriptor.required_features |= required_features;
    pollster::block_on(adapter.request_device(&device_descriptor))
        .unwrap_or_else(|error| panic!("cannot create device for {requirement}: {error}"))
}
