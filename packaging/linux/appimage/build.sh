#!/bin/sh
set -eu

APPIMAGE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$APPIMAGE_DIR/../../.." && pwd)
. "$APPIMAGE_DIR/versions.env"

TARGET_DIR=${CARGO_TARGET_DIR:-"$ROOT/target"}
TOOL=${APPIMAGETOOL:-"$TARGET_DIR/tools/appimagetool"}
RUNTIME=${APPIMAGE_RUNTIME:-"$TARGET_DIR/tools/runtime-x86_64"}
DEFAULT_OUTPUT="$TARGET_DIR/appimage/xl-view-x86_64.AppImage"

download_verified() {
    url=$1
    checksum=$2
    destination=$3
    mode=$4

    if [ -f "$destination" ]; then
        echo "$checksum  $destination" | sha256sum --check --strict
        return
    fi

    temporary="${destination}.download"
    install -d "$(dirname -- "$destination")"
    curl --fail --location --show-error "$url" --output "$temporary"
    if ! echo "$checksum  $temporary" | sha256sum --check --strict; then
        rm -f "$temporary"
        exit 1
    fi
    chmod "$mode" "$temporary"
    mv "$temporary" "$destination"
}

assemble_appdir() {
    binary=$1
    appdir=$2

    if [ ! -x "$binary" ]; then
        echo "xl-view binary is missing or not executable: $binary" >&2
        exit 1
    fi
    if [ -e "$appdir" ]; then
        echo "refusing to reuse staging directory: $appdir" >&2
        exit 1
    fi

    install -d \
        "$appdir/usr/bin" \
        "$appdir/usr/share/applications" \
        "$appdir/usr/share/icons/hicolor/512x512/apps" \
        "$appdir/usr/share/icons/hicolor/scalable/apps" \
        "$appdir/usr/share/licenses/xl-view" \
        "$appdir/usr/share/metainfo"
    install -m 0755 "$binary" "$appdir/usr/bin/xl-view"
    install -m 0755 "$APPIMAGE_DIR/AppRun" "$appdir/AppRun"
    install -m 0644 "$ROOT/packaging/linux/io.github.andrinbr.xl_view.desktop" \
        "$appdir/usr/share/applications/io.github.andrinbr.xl_view.desktop"
    install -m 0644 "$ROOT/packaging/linux/io.github.andrinbr.xl_view.appdata.xml" \
        "$appdir/usr/share/metainfo/io.github.andrinbr.xl_view.appdata.xml"
    install -m 0644 "$ROOT/assets/icons/xl-view.png" \
        "$appdir/usr/share/icons/hicolor/512x512/apps/xl-view.png"
    install -m 0644 "$ROOT/assets/icons/xl-view.svg" \
        "$appdir/usr/share/icons/hicolor/scalable/apps/xl-view.svg"
    install -m 0644 "$ROOT/COPYRIGHT" \
        "$appdir/usr/share/licenses/xl-view/COPYRIGHT"
    install -m 0644 "$ROOT/LICENSE" \
        "$appdir/usr/share/licenses/xl-view/LICENSE"
    install -m 0644 "$ROOT/THIRD-PARTY-LICENSES.html" \
        "$appdir/usr/share/licenses/xl-view/THIRD-PARTY-LICENSES.html"
    install -m 0644 "$ROOT/assets/fonts/AdwaitaSans-LICENSE.txt" \
        "$appdir/usr/share/licenses/xl-view/AdwaitaSans-LICENSE.txt"

    ln -s usr/share/applications/io.github.andrinbr.xl_view.desktop \
        "$appdir/io.github.andrinbr.xl_view.desktop"
    ln -s usr/share/icons/hicolor/512x512/apps/xl-view.png "$appdir/xl-view.png"
    ln -s xl-view.png "$appdir/.DirIcon"
}

case ${1:-} in
    -*)
        echo "unknown option: $1" >&2
        exit 2
        ;;
    *)
        # Non-option arguments are output paths handled below.
        :
        ;;
esac
if [ "$#" -gt 1 ]; then
    echo "usage: $0 [output.AppImage]" >&2
    exit 2
fi

output=${1:-$DEFAULT_OUTPUT}
staging_root=$(mktemp -d "${TMPDIR:-/tmp}/xl-view-appimage.XXXXXX")
appdir="$staging_root/xl-view.AppDir"
trap 'rm -rf "$staging_root"' EXIT HUP INT TERM

cargo build --manifest-path "$ROOT/Cargo.toml" --release --locked
assemble_appdir "$TARGET_DIR/release/xl-view" "$appdir"
"$APPIMAGE_DIR/audit-appdir.sh" "$appdir"

download_verified "$APPIMAGETOOL_URL" "$APPIMAGETOOL_SHA256" "$TOOL" 0755
download_verified "$APPIMAGE_RUNTIME_URL" "$APPIMAGE_RUNTIME_SHA256" "$RUNTIME" 0644
mkdir -p "$(dirname -- "$output")"

ARCH=x86_64 VERSION="$(cargo metadata --manifest-path "$ROOT/Cargo.toml" --no-deps --format-version 1 | sed -n 's/.*"version":"\([^"]*\)".*/\1/p')" \
APPIMAGE_EXTRACT_AND_RUN=1 "$TOOL" --no-appstream --runtime-file "$RUNTIME" "$appdir" "$output"
printf '%s\n' "$output"
