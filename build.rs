// Icon embedded in the Windows executable.
const ICON_PATH: &str = "assets/icons/io.github.andrinbr.xl_view.ico";

fn main() {
    // Rebuild the Windows resources when the icon changes.
    println!("cargo::rerun-if-changed={ICON_PATH}");

    // Skip Windows resource compilation for other targets.
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "windows" {
        return;
    }

    // Compile and link the icon and metadata from Cargo.toml into the executable.
    let mut resource = winresource::WindowsResource::new();
    resource.set_icon(ICON_PATH);
    resource.compile().unwrap_or_else(|error| {
        panic!("failed to compile Windows resources from {ICON_PATH}: {error}")
    });
}
