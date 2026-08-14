#!/bin/sh
set -eu

# Use the fixed paths provided by the builder image.
APPLICATION_ID=io.github.andrinbr.xl_view
OUTPUT_DIR=/work/target/appimage
PACKAGER_CONFIG=packaging/linux/Packager.toml
release_dir=${CARGO_TARGET_DIR:?CARGO_TARGET_DIR must be set by the builder image}/release

# Create an isolated workspace for packaging and inspection.
staging_root=$(mktemp -d)
trap 'rm -rf "$staging_root"' EXIT HUP INT TERM
package_dir="$staging_root/package"
audit_dir="$staging_root/audit"
mkdir -p "$package_dir" "$audit_dir"

# Compile the release binary into the mounted target directory.
cd /work
cargo build --release --locked

# Stage the binary and desktop-ID alias where cargo-packager expects them.
install -m 0755 \
    "$release_dir/xl-view" \
    "$package_dir/xl-view"
ln -f \
    "$package_dir/xl-view" \
    "$package_dir/$APPLICATION_ID.bin"

# Package the binaries and metadata as an AppImage.
APPIMAGE_EXTRACT_AND_RUN=1 cargo packager \
    --config "$PACKAGER_CONFIG" \
    --out-dir "$package_dir"

# Select the single AppImage produced by cargo-packager.
set -- "$package_dir"/*.AppImage
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "cargo-packager did not produce exactly one AppImage" >&2
    exit 1
fi
appimage=$1

# Extract the finished package and validate its important integration points.
(
    cd "$audit_dir"
    "$appimage" --appimage-extract >/dev/null
)

# Locate the packaged files used by the audit.
appdir="$audit_dir/squashfs-root"
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
dynamic_info="$audit_dir/dynamic.txt"
readelf -d "$binary" > "$dynamic_info"
if grep -qE 'NEEDED.*(libX11|libGL|libEGL|libGLES|libvulkan|libwayland)' "$dynamic_info"; then
    echo "binary has a forbidden direct display/GPU dependency" >&2
    exit 1
fi

library_paths="$audit_dir/libraries.txt"
find "$appdir" -name '*.so*' > "$library_paths"
if grep -qE '/(libX11|libGL|libEGL|libGLES|libvulkan|libwayland|libdrm|libgbm)[^/]*\.so' "$library_paths"; then
    echo "AppImage bundles a forbidden host display/GPU library" >&2
    exit 1
fi

if grep -aqE 'VK_(ICD_FILENAMES|LAYER_PATH)' "$appdir/AppRun"; then
    echo "AppRun overrides the host Vulkan loader configuration" >&2
    exit 1
fi

# The binary's --version reflects Cargo.toml. The Packager config and the
# AppStream release entry must agree with it.
version=$(APPIMAGE_EXTRACT_AND_RUN=1 "$appimage" --version)
version=${version#xl-view }
grep -Fqx "version = \"$version\"" "$PACKAGER_CONFIG" \
    || { echo "$PACKAGER_CONFIG version does not match xl-view $version" >&2; exit 1; }
grep -Fq "<release version=\"$version\"" "$appdata" \
    || { echo "$appdata has no <release version=\"$version\"> entry" >&2; exit 1; }

# Publish only the audited AppImage to the mounted source tree.
OUTPUT="$OUTPUT_DIR/xl-view-v${version}-linux-x86_64.AppImage"
mkdir -p "$OUTPUT_DIR"
install -m 0755 "$appimage" "$OUTPUT"
printf '%s\n' "AppImage built and audited: $OUTPUT"
