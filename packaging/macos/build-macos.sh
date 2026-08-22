#!/usr/bin/env bash
set -eu

# Resolve all paths from the script location.
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd)
packager_config="$script_dir/Packager.toml"
output_dir="$repository_root/target/macos"
target_triple=aarch64-apple-darwin
release_binary="$repository_root/target/$target_triple/release/xl-view"

# Create an isolated workspace and detach a mounted audit image on failure.
staging_root=$(mktemp -d)
audit_mount=
cleanup() {
    if [[ -n "$audit_mount" ]]; then
        hdiutil detach "$audit_mount" >/dev/null 2>&1 || true
    fi
    rm -rf "$staging_root"
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

# Report clear failures for required files and expected values.
assert_nonempty_file() {
    local file_path="$1"

    if [[ ! -s "$file_path" ]]; then
        echo "$file_path is missing or empty" >&2
        exit 1
    fi
}

assert_equal() {
    local actual="$1"
    local expected="$2"
    local description="$3"

    if [[ "$actual" != "$expected" ]]; then
        echo "$description is '$actual'; expected '$expected'" >&2
        exit 1
    fi
}

# Create the directory cargo-packager will use for inputs and outputs.
package_dir="$staging_root/package"
mkdir -p "$package_dir"

# Compile the Apple silicon release binary.
cd "$repository_root"
cargo build --release --locked --target "$target_triple"

# Ensure Cargo.toml and the standalone Packager config use the same version.
version_output=$("$release_binary" --version)
version=${version_output#xl-view }
assert_equal "$version_output" "xl-view $version" "release binary version"
grep -Fqx "version = \"$version\"" "$packager_config" \
    || { echo "$packager_config version does not match xl-view $version" >&2; exit 1; }

# Stage the binary and package it without Finder automation.
install -m 0755 "$release_binary" "$package_dir/xl-view"
CI=true cargo packager \
    --config "$packager_config" \
    --out-dir "$package_dir" \
    --target "$target_triple"

# Select and verify the single DMG produced by cargo-packager.
set -- "$package_dir"/*.dmg
if [[ "$#" -ne 1 ]] || [[ ! -f "$1" ]]; then
    echo "cargo-packager did not produce exactly one DMG" >&2
    exit 1
fi
dmg=$1
hdiutil verify "$dmg" >/dev/null

# Mount the finished DMG so the shipped app can be audited directly.
audit_mount="$staging_root/mounted-dmg"
hdiutil attach -readonly -nobrowse -mountpoint "$audit_mount" "$dmg" >/dev/null

# Locate the packaged files used by the audit.
app_bundle="$audit_mount/XL-View.app"
contents="$app_bundle/Contents"
plist="$contents/Info.plist"
packaged_binary="$contents/MacOS/xl-view"
resources="$contents/Resources"

# Validate the application bundle metadata.
assert_equal "$(plutil -extract CFBundleIdentifier raw "$plist")" \
    "io.github.andrinbr.xl-view" "bundle identifier"
assert_equal "$(plutil -extract CFBundleExecutable raw "$plist")" \
    "xl-view" "bundle executable"
assert_equal "$(plutil -extract CFBundleShortVersionString raw "$plist")" \
    "$version" "bundle version"

# Check the icon and license payloads.
icon_file=$(plutil -extract CFBundleIconFile raw "$plist")
assert_nonempty_file "$resources/$icon_file"
for license_file in LICENSE COPYRIGHT THIRD-PARTY-LICENSES.html AdwaitaSans-LICENSE.txt; do
    assert_nonempty_file "$resources/licenses/$license_file"
done

# Validate the architecture and version of the binary that will ship.
assert_equal "$(lipo -archs "$packaged_binary")" "arm64" "packaged binary architecture"
assert_equal "$("$packaged_binary" --version)" "xl-view $version" "packaged binary version"

# Detach the audited image before copying its backing file.
hdiutil detach "$audit_mount" >/dev/null
audit_mount=

# Publish only the verified and audited DMG.
artifact="$output_dir/xl-view-v${version}-macos-aarch64.dmg"
mkdir -p "$output_dir"
install -m 0644 "$dmg" "$artifact"
printf '%s\n' "macOS package built and audited: $artifact"
