#!/bin/sh
set -eu

# Use the fixed paths provided by the builder image.
APPLICATION_ID=io.github.andrinbr.xl_view
OUTPUT=/work/target/appimage/xl-view-x86_64.AppImage
release_dir=${CARGO_TARGET_DIR:?CARGO_TARGET_DIR must be set by the builder image}/release

# Create an isolated workspace for packaging and inspection.
staging_root=$(mktemp -d)
trap 'rm -rf "$staging_root"' EXIT HUP INT TERM

# Compile the release binary into the mounted target directory.
cd /work
cargo build --release --locked

# cargo-packager uses the main binary name for the desktop file name.
ln -f \
    "$release_dir/xl-view" \
    "$release_dir/$APPLICATION_ID.bin"

# Package the binaries and metadata as an AppImage.
APPIMAGE_EXTRACT_AND_RUN=1 cargo packager \
    --release \
    --formats appimage \
    --out-dir "$staging_root"

# Select the single AppImage produced by cargo-packager.
set -- "$staging_root"/*.AppImage
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "cargo-packager did not produce exactly one AppImage" >&2
    exit 1
fi
appimage=$1

# Extract the finished package and validate its important integration points.
(
    cd "$staging_root"
    "$appimage" --appimage-extract >/dev/null
)

# Locate the packaged files used by the audit.
appdir="$staging_root/squashfs-root"
binary="$appdir/usr/bin/xl-view"
desktop="$appdir/usr/share/applications/$APPLICATION_ID.desktop"
appdata="$appdir/usr/share/metainfo/$APPLICATION_ID.appdata.xml"
license_dir="$appdir/usr/share/licenses/xl-view"

# Check AppImage integration files and license payloads.
test -e "$appdir/$APPLICATION_ID.desktop"
test -f "$appdir/.DirIcon"
test -f "$appdir/usr/share/icons/hicolor/512x512/apps/$APPLICATION_ID.png"
test -f "$appdir/usr/share/icons/hicolor/scalable/apps/$APPLICATION_ID.svg"
test -s "$license_dir/COPYRIGHT"
test -s "$license_dir/LICENSE"
test -s "$license_dir/THIRD-PARTY-LICENSES.html"
test -s "$license_dir/AdwaitaSans-LICENSE.txt"
test -s "$license_dir/AppImage-RUNTIME-LICENSE.txt"

# Validate desktop and AppStream metadata.
desktop-file-validate "$desktop"
appstreamcli validate --no-net "$appdata"
grep -Fqx "Exec=xl-view %f" "$desktop"
grep -Fqx "Icon=$APPLICATION_ID" "$desktop"
grep -Fq "<launchable type=\"desktop-id\">$APPLICATION_ID.desktop</launchable>" "$appdata"

# Display and GPU libraries must come from the host.
dynamic_info="$staging_root/dynamic.txt"
readelf -d "$binary" > "$dynamic_info"
if grep -qE 'NEEDED.*(libX11|libGL|libEGL|libGLES|libvulkan|libwayland)' "$dynamic_info"; then
    echo "binary has a forbidden direct display/GPU dependency" >&2
    exit 1
fi

library_paths="$staging_root/libraries.txt"
find "$appdir" -name '*.so*' > "$library_paths"
if grep -qE '/(libX11|libGL|libEGL|libGLES|libvulkan|libwayland|libdrm|libgbm)[^/]*\.so' "$library_paths"; then
    echo "AppImage bundles a forbidden host display/GPU library" >&2
    exit 1
fi

if grep -aqE 'VK_(ICD_FILENAMES|LAYER_PATH)' "$appdir/AppRun"; then
    echo "AppRun overrides the host Vulkan loader configuration" >&2
    exit 1
fi

# Confirm that the executable and AppStream versions agree.
version_output=$(APPIMAGE_EXTRACT_AND_RUN=1 "$appimage" --version)
case "$version_output" in
    "xl-view "*) version=${version_output#xl-view } ;;
    *) echo "unexpected version output: $version_output" >&2; exit 1 ;;
esac
grep -Fq "<release version=\"$version\"" "$appdata"

# Publish only the audited AppImage to the mounted source tree.
mkdir -p /work/target/appimage
install -m 0755 "$appimage" "$OUTPUT"
printf '%s\n' "AppImage built and audited: $OUTPUT"
