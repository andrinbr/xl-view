# AppImage

The AppImage contains xl-view, its application metadata, and its license
notices. The host provides Wayland, Vulkan, and the GPU driver. License files
are installed under `usr/share/licenses/xl-view/` inside the AppImage.

## Build

Run the build from the repository root:

```console
packaging/linux/appimage/build.sh
target/appimage/xl-view-x86_64.AppImage --appimage-extract-and-run --version
```

The first command builds and checks the package. It downloads pinned,
checksum-verified AppImage tools into `target/tools` when needed. Pass an output
path to write the AppImage somewhere else:

```console
packaging/linux/appimage/build.sh path/to/xl-view.AppImage
```

## Container build

The container provides the pinned Debian/glibc build environment used by CI.
From the repository root, run:

```console
buildah build --file packaging/linux/appimage/Dockerfile --tag xl-view-appimage .
podman run --rm --userns=keep-id --volume "$PWD:/work:Z" xl-view-appimage
```

The AppImage is written to `target/appimage/`.
