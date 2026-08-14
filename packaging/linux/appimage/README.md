# AppImage

The AppImage contains xl-view, its application metadata, and its license
notices. The host provides Wayland, Vulkan, and the GPU driver. License files
are installed under `usr/share/licenses/xl-view/` inside the AppImage.

## Build

The builder image provides the pinned Debian/glibc environment used by CI,
avoiding host-dependent compatibility differences. From the repository root,
run:

```console
buildah build --file packaging/linux/appimage/Dockerfile --tag xl-view-appimage packaging/linux/appimage
podman run --rm --userns=keep-id:uid=1000,gid=1000 --volume "$PWD:/work:Z" xl-view-appimage
```

The container compiles, packages, and audits the mounted source tree. The result
is written to `target/appimage/xl-view-vX.Y.Z-linux-x86_64.AppImage`.

Verify that it starts with:

```console
target/appimage/xl-view-vX.Y.Z-linux-x86_64.AppImage --version
```
