# macOS package

The macOS release is an unsigned Apple silicon (ARM64) DMG containing
`XL-View.app`.

With the configured Rust toolchain available, run these commands from the
repository root:

```console
cargo install cargo-packager --version 0.11.8 --locked
./packaging/macos/build-macos.sh
```

The script compiles the ARM64 release binary, creates and verifies the DMG, then
mounts it to check the bundle metadata, icon, licenses, architecture, and
packaged version. The result is written to:

```text
target/macos/xl-view-vX.Y.Z-macos-aarch64.dmg
```
