#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
fixtures="$root/tests/fixtures"
sources="$fixtures/sources"
work=$(mktemp -d)
trap 'rm -rf "$work"' 0
trap 'exit 1' HUP INT TERM

# HDR test pattern.
#
# The RGB sample values are identical for all three outputs. Only the tagged
# colour space and transfer function differ.
#
# Layout:
#   y=0..400     Eight grayscale patches
#   y=401..511   Constant gray
#   y=512..575   Black
#   y=576..639   Blue ramp
#   y=640..703   Green ramp
#   y=704..767   Cyan ramp
#   y=768..831   Red ramp
#   y=832..895   Magenta ramp
#   y=896..959   Yellow ramp
#   y=960..1023  Gray ramp
q='i/(w-1)'
bar='(floor(i/128)*128+127)/(w-1)'
gray='38437/65535'
common="j<=400 ? $bar : j<512 ? $gray : j<576 ? 0 :"

# Create the ppm file
magick -size 1024x1024 xc:black \
    -colorspace RGB \
    -depth 16 \
    -channel R \
    -fx "$common (j>=768 ? $q : 0)" \
    -channel G \
    -fx "$common (((j>=640 && j<768) || j>=896) ? $q : 0)" \
    -channel B \
    -fx "$common (
        ((j>=576 && j<640) ||
         (j>=704 && j<768) ||
         (j>=832 && j<896) ||
         j>=960) ? $q : 0
    )" \
    +channel \
    "$work/test-pattern.ppm"

# Encode the testpattern with each transfer function and color space.
cjxl "$work/test-pattern.ppm" "$fixtures/test_pattern-sRGB.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    -x color_space=sRGB --quiet
cjxl "$work/test-pattern.ppm" "$fixtures/test_pattern-HLG.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    --intensity_target=1000 -x color_space=Rec2100HLG --quiet
cjxl "$work/test-pattern.ppm" "$fixtures/test_pattern-PQ.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    --intensity_target=10000 -x color_space=Rec2100PQ --quiet

# A linear-in-luminance grayscale ramp. PQ code values are absolute; HLG code
# values include the nominal display OOTF for the selected peak.
magick -size 1024x64 xc:black -colorspace RGB -channel RGB \
    -fx 'pow((3424.0/4096.0+(2413.0/128.0)*pow((1000.0/10000.0)*i/(w-1),2610.0/16384.0))/(1.0+(2392.0/128.0)*pow((1000.0/10000.0)*i/(w-1),2610.0/16384.0)),2523.0/32.0)' \
    -depth 16 "$work/ramp-pq-1000.ppm"
magick -size 1024x64 xc:black -colorspace RGB -channel RGB \
    -fx 'pow((3424.0/4096.0+(2413.0/128.0)*pow(i/(w-1),2610.0/16384.0))/(1.0+(2392.0/128.0)*pow(i/(w-1),2610.0/16384.0)),2523.0/32.0)' \
    -depth 16 "$work/ramp-pq-10000.ppm"
magick -size 1024x64 xc:black -colorspace RGB -channel RGB \
    -fx '(pow(i/(w-1),1.0/1.2)<=1.0/12.0)?sqrt(3.0*pow(i/(w-1),1.0/1.2)):0.17883277*ln(12.0*pow(i/(w-1),1.0/1.2)-0.28466892)+0.55991073' \
    -depth 16 "$work/ramp-hlg-1000.ppm"
magick -size 1024x64 xc:black -colorspace RGB -channel RGB \
    -fx '(pow(i/(w-1),1.0/(1.2+0.42*log(2.0)/log(10.0)))<=1.0/12.0)?sqrt(3.0*pow(i/(w-1),1.0/(1.2+0.42*log(2.0)/log(10.0)))):0.17883277*ln(12.0*pow(i/(w-1),1.0/(1.2+0.42*log(2.0)/log(10.0)))-0.28466892)+0.55991073' \
    -depth 16 "$work/ramp-hlg-2000.ppm"

# Encode each ramp with its matching HDR transfer function and peak brightness.
cjxl "$work/ramp-hlg-1000.ppm" "$fixtures/ramp-hlg-1000.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    --intensity_target=1000 -x color_space=Rec2100HLG --quiet
cjxl "$work/ramp-hlg-2000.ppm" "$fixtures/ramp-hlg-2000.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    --intensity_target=2000 -x color_space=Rec2100HLG --quiet
cjxl "$work/ramp-pq-1000.ppm" "$fixtures/ramp-pq-1000.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    --intensity_target=1000 -x color_space=Rec2100PQ --quiet
cjxl "$work/ramp-pq-10000.ppm" "$fixtures/ramp-pq-10000.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    --intensity_target=10000 -x color_space=Rec2100PQ --quiet

# Check libjxl's transfer-function-specific intensity defaults by omitting
# --intensity_target entirely: 1,000 nits for HLG and 10,000 nits for PQ.
cjxl "$work/ramp-hlg-1000.ppm" "$fixtures/ramp-hlg-no-intensity.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    -x color_space=Rec2100HLG --quiet
cjxl "$work/ramp-pq-10000.ppm" "$fixtures/ramp-pq-no-intensity.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    -x color_space=Rec2100PQ --quiet

# Plain single-channel grayscale, no color transform or ICC profile involved.
magick "$sources/grayscale.pgm" "$work/grayscale.png"
cjxl "$work/grayscale.png" "$fixtures/grayscale.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 --quiet

# Same source image encoded three ways: linear BT.2020 primaries, then PQ and
# HLG, to check color-space tagging independent of the transfer function.
magick "$sources/rgb.ppm" "$work/rgb.ppm"
cjxl "$work/rgb.ppm" "$fixtures/linear-bt2020.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    -x color_space=RGB_D65_202_Rel_Lin --quiet
cjxl "$work/rgb.ppm" "$fixtures/reference-pq.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    --intensity_target=10000 -x color_space=Rec2100PQ --quiet
cjxl "$work/rgb.ppm" "$fixtures/reference-hlg.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 \
    --intensity_target=1000 -x color_space=Rec2100HLG --quiet

# A JPEG with an EXIF Orientation tag, transcoded (not lossless-JPEG-recoded)
# so the pixel data is re-encoded in the requested orientation.
cjpeg -lossless 1 -rgb -optimize \
    -outfile "$work/oriented-base.jpg" "$work/rgb.ppm"
exif --create-exif --output="$work/oriented.jpg" --ifd=0 --tag=Orientation \
    --set-value=6 --no-fixup "$work/oriented-base.jpg"
cjxl "$work/oriented.jpg" "$fixtures/oriented.jxl" \
    --lossless_jpeg=0 --distance=0 --modular=1 --effort=3 \
    --num_threads=1 --quiet

# Straight (non-premultiplied) alpha: opaque, half-transparent, and fully
# transparent pixels per channel, including a fully transparent black pixel.
magick -size 3x2 xc:none \
    -fill 'rgba(255,0,0,1)' -draw 'point 0,0' \
    -fill 'rgba(0,255,0,0.5)' -draw 'point 1,0' \
    -fill 'rgba(0,0,255,0)' -draw 'point 2,0' \
    -fill 'rgba(255,255,255,1)' -draw 'point 0,1' \
    -fill 'rgba(128,128,128,0.5)' -draw 'point 1,1' \
    -fill 'rgba(0,0,0,0)' -draw 'point 2,1' \
    "$work/alpha.png"
cjxl "$work/alpha.png" "$fixtures/alpha.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 --quiet

# Same layout as above, but colors are pre-multiplied by alpha and encoded
# with --premultiply=1, to test associated-alpha handling separately.
magick -size 3x2 xc:none \
    -fill 'rgba(255,0,0,1)' -draw 'point 0,0' \
    -fill 'rgba(0,128,0,0.5)' -draw 'point 1,0' \
    -fill 'rgba(0,0,0,0)' -draw 'point 2,0' \
    -fill 'rgba(255,255,255,1)' -draw 'point 0,1' \
    -fill 'rgba(64,64,64,0.5)' -draw 'point 1,1' \
    -fill 'rgba(0,0,0,0)' -draw 'point 2,1' \
    "$work/alpha-associated.png"
cjxl "$work/alpha-associated.png" "$fixtures/associated-alpha.jxl" \
    --distance=0 --modular=1 --premultiply=1 --effort=3 --num_threads=1 --quiet

# Non-standard embedded ICC profile, to test that we preserve/read a profile
# libjxl doesn't have a built-in enum for instead of falling back to sRGB.
# Path comes from the colord-data (colord on fedora) package.
adobe_rgb=/usr/share/color/icc/colord/AdobeRGB1998.icc
magick "$sources/rgb.ppm" -profile "$adobe_rgb" "$work/rgb-adobe.png"
cjxl "$work/rgb-adobe.png" "$fixtures/icc.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 --quiet

# Two-frame animation, to exercise the animated/multi-frame decode path.
magick -size 2x2 xc:red -delay 10 -size 2x2 xc:blue -loop 0 "$work/animation.gif"
cjxl "$work/animation.gif" "$fixtures/animation.jxl" \
    --distance=0 --modular=1 --effort=3 --num_threads=1 --quiet

# Helper: apply one EXIF tag to exif-current.jpg and replace it in place, so
# the calls below read as a flat list of tags rather than a pipeline.
set_exif_tag() {
    exif --output="$work/exif-next.jpg" --ifd="$1" --tag="$2" \
        --set-value="$3" --no-fixup "$work/exif-current.jpg"
    mv "$work/exif-next.jpg" "$work/exif-current.jpg"
}

# JPEG carrying a broad set of common EXIF tags (camera info, exposure, lens,
# free-text fields), lossless-JPEG-recoded into JXL.
magick "$sources/rgb.ppm" -quality 95 "$work/exif-base.jpg"
exif --create-exif --output="$work/exif-current.jpg" --ifd=0 --tag=Make \
    --set-value=ACME --no-fixup "$work/exif-base.jpg"
set_exif_tag 0 Model 'Photon 1'
set_exif_tag 0 Software 'Fixture Maker 1.0'
set_exif_tag 0 Artist 'Ada Example'
set_exif_tag 0 Copyright 'CC0 fixture'
set_exif_tag EXIF ExposureTime '1 125'
set_exif_tag EXIF FNumber '28 10'
set_exif_tag EXIF ISOSpeedRatings 200
set_exif_tag EXIF DateTimeOriginal '2026:07:13 12:34:56'
set_exif_tag EXIF ExposureBiasValue '-1 3'
set_exif_tag EXIF FocalLength '50 1'
set_exif_tag EXIF LensMake 'ACME Optics'
set_exif_tag EXIF LensModel 'Prime 50'
set_exif_tag EXIF UserComment 'retained unparsed field'
cjxl "$work/exif-current.jpg" "$fixtures/exif-common.jxl" \
    --lossless_jpeg=1 --effort=3 --num_threads=1 --quiet

# JPEG carrying only an XMP Rating, to test XMP extraction independent of
# the EXIF-heavy fixtures above.
cp "$work/exif-base.jpg" "$work/xmp-rating.jpg"
exiv2 -M 'set Xmp.xmp.Rating 4' "$work/xmp-rating.jpg"
cjxl "$work/xmp-rating.jpg" "$fixtures/xmp-rating.jxl" \
    --lossless_jpeg=1 --effort=3 --num_threads=1 --quiet

generated_fixtures='test_pattern-sRGB.jxl
test_pattern-HLG.jxl
test_pattern-PQ.jxl
ramp-hlg-1000.jxl
ramp-hlg-2000.jxl
ramp-pq-1000.jxl
ramp-pq-10000.jxl
ramp-hlg-no-intensity.jxl
ramp-pq-no-intensity.jxl
grayscale.jxl
linear-bt2020.jxl
reference-pq.jxl
reference-hlg.jxl
oriented.jxl
alpha.jxl
associated-alpha.jxl
icc.jxl
animation.jxl
exif-common.jxl
xmp-rating.jxl'

# Print each fixture's metadata for manual review before committing bytes.
for fixture in $generated_fixtures
do
    jxlinfo "$fixtures/$fixture"
done

# Print checksums so reviewers can confirm byte-for-byte reproducibility.
for fixture in $generated_fixtures
do
    sha256sum "$fixtures/$fixture"
done
