#!/bin/sh
set -eu

APPIMAGE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$APPIMAGE_DIR/../../.." && pwd)
CALLER_DIR=$(pwd)

TARGET_DIR=${CARGO_TARGET_DIR:-"$ROOT/target"}
case $TARGET_DIR in
    /*) ;;
    *) TARGET_DIR="$ROOT/$TARGET_DIR" ;;
esac

DEFAULT_OUTPUT="$ROOT/target/appimage/xl-view-x86_64.AppImage"
APPLICATION_ID=io.github.andrinbr.xl_view

case ${1:-} in
    -*)
        echo "unknown option: $1" >&2
        exit 2
        ;;
esac
if [ "$#" -gt 1 ]; then
    echo "usage: $0 [output.AppImage]" >&2
    exit 2
fi

output=${1:-$DEFAULT_OUTPUT}
case $output in
    /*) ;;
    *) output="$CALLER_DIR/$output" ;;
esac

package_root=$(mktemp -d "${TMPDIR:-/tmp}/xl-view-packager.XXXXXX")
trap 'rm -rf "$package_root"' EXIT HUP INT TERM

cd "$ROOT"
cargo build --release --locked

release_dir="$TARGET_DIR/release"
source_binary="$release_dir/xl-view"
packager_main="$release_dir/$APPLICATION_ID.bin"
if [ ! -x "$source_binary" ]; then
    echo "xl-view binary is missing or not executable: $source_binary" >&2
    exit 1
fi
ln -f "$source_binary" "$packager_main"

APPIMAGE_EXTRACT_AND_RUN=1 cargo packager \
    --release \
    --formats appimage \
    --out-dir "$package_root"

set -- "$package_root"/*.AppImage
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "cargo-packager did not produce exactly one AppImage" >&2
    exit 1
fi

mkdir -p "$(dirname -- "$output")"
install -m 0755 "$1" "$output"
"$APPIMAGE_DIR/audit-appimage.sh" "$output"
printf '%s\n' "$output"
