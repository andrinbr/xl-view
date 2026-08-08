#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <AppImage>" >&2
    exit 2
fi

case $1 in
    /*) APPIMAGE=$1 ;;
    *) APPIMAGE="$(pwd)/$1" ;;
esac
if [ ! -x "$APPIMAGE" ]; then
    echo "AppImage is missing or not executable: $APPIMAGE" >&2
    exit 1
fi

APPLICATION_ID=io.github.andrinbr.xl_view
staging_root=$(mktemp -d "${TMPDIR:-/tmp}/xl-view-appimage-audit.XXXXXX")
trap 'rm -rf "$staging_root"' EXIT HUP INT TERM

(
    cd "$staging_root"
    "$APPIMAGE" --appimage-extract >/dev/null
)

APPDIR="$staging_root/squashfs-root"
BINARY="$APPDIR/usr/bin/xl-view"
DESKTOP="$APPDIR/usr/share/applications/$APPLICATION_ID.desktop"
APPDATA="$APPDIR/usr/share/metainfo/$APPLICATION_ID.appdata.xml"
LICENSE_DIR="$APPDIR/usr/share/licenses/xl-view"

test -x "$APPDIR/AppRun"
test -x "$BINARY"
test -f "$DESKTOP"
test -e "$APPDIR/$APPLICATION_ID.desktop"
test -f "$APPDATA"
test -f "$APPDIR/.DirIcon"
test -f "$APPDIR/usr/share/icons/hicolor/512x512/apps/$APPLICATION_ID.png"
test -f "$APPDIR/usr/share/icons/hicolor/scalable/apps/$APPLICATION_ID.svg"
test -s "$LICENSE_DIR/COPYRIGHT"
test -s "$LICENSE_DIR/LICENSE"
test -s "$LICENSE_DIR/THIRD-PARTY-LICENSES.html"
test -s "$LICENSE_DIR/AdwaitaSans-LICENSE.txt"

desktop-file-validate "$DESKTOP"
appstreamcli validate --no-net "$APPDATA"
grep -Fqx "Exec=xl-view %f" "$DESKTOP"
grep -Fqx "Icon=$APPLICATION_ID" "$DESKTOP"
grep -Fqx "StartupNotify=true" "$DESKTOP"
grep -Fqx "StartupWMClass=$APPLICATION_ID" "$DESKTOP"
grep -Fq "<launchable type=\"desktop-id\">$APPLICATION_ID.desktop</launchable>" "$APPDATA"

if readelf -d "$BINARY" | grep -E 'NEEDED.*(libX11|libGL|libEGL|libGLES|libvulkan|libwayland)' >/dev/null; then
    echo "binary has a forbidden direct display/GPU dependency" >&2
    exit 1
fi

if find "$APPDIR" -type f | grep -E '/(libX11|libGL|libEGL|libGLES|libvulkan|libwayland|libdrm|libgbm)[^/]*\.so' >/dev/null; then
    echo "AppImage bundles a forbidden host display/GPU library" >&2
    exit 1
fi

if find "$APPDIR/usr/lib" "$APPDIR/usr/lib64" -mindepth 1 -print -quit | grep -q .; then
    echo "AppImage must not bundle libraries that can override host libraries" >&2
    exit 1
fi

if grep -aE 'VK_(ICD_FILENAMES|LAYER_PATH)' "$APPDIR/AppRun" >/dev/null; then
    echo "AppRun overrides the host Vulkan loader configuration" >&2
    exit 1
fi

version_output=$("$APPIMAGE" --appimage-extract-and-run --version)
case $version_output in
    "xl-view "*) ;;
    *)
        echo "unexpected AppImage version output: $version_output" >&2
        exit 1
        ;;
esac
version=${version_output#xl-view }
grep -Fq "<release version=\"$version\"" "$APPDATA"

printf '%s\n' "AppImage audit passed: $APPIMAGE"
