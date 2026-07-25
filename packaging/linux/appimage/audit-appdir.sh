#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <AppDir>" >&2
    exit 2
fi

APPDIR=$1
BINARY="$APPDIR/usr/bin/xl-view"
DESKTOP="$APPDIR/usr/share/applications/io.github.andrinbr.xl_view.desktop"
APPDATA="$APPDIR/usr/share/metainfo/io.github.andrinbr.xl_view.appdata.xml"
LICENSE_DIR="$APPDIR/usr/share/licenses/xl-view"

test -x "$APPDIR/AppRun"
test -x "$BINARY"
test -f "$DESKTOP"
test -f "$APPDATA"
test -f "$APPDIR/usr/share/icons/hicolor/512x512/apps/xl-view.png"
test -f "$APPDIR/usr/share/icons/hicolor/scalable/apps/xl-view.svg"
test -s "$LICENSE_DIR/COPYRIGHT"
test -s "$LICENSE_DIR/LICENSE"
test -s "$LICENSE_DIR/THIRD-PARTY-LICENSES.html"
test -s "$LICENSE_DIR/AdwaitaSans-LICENSE.txt"

desktop-file-validate "$DESKTOP"
appstreamcli validate --no-net "$APPDATA"

if readelf -d "$BINARY" | grep -E 'NEEDED.*(libX11|libGL|libEGL|libGLES|libvulkan|libwayland)' >/dev/null; then
    echo "binary has a forbidden direct display/GPU dependency" >&2
    exit 1
fi

if find "$APPDIR" -type f | grep -E '/(libX11|libGL|libEGL|libGLES|libvulkan|libwayland|libdrm|libgbm)[^/]*\.so' >/dev/null; then
    echo "AppDir bundles a forbidden host display/GPU library" >&2
    exit 1
fi

if grep -E 'VK_(ICD_FILENAMES|LAYER_PATH)|LD_LIBRARY_PATH' "$APPDIR/AppRun" >/dev/null; then
    echo "AppRun overrides the host loader configuration" >&2
    exit 1
fi

printf '%s\n' "AppDir audit passed: $APPDIR"
