# Test fixtures

The checked-in JPEG XL files are inputs for decoder integration tests. The
`sources/` directory contains small PGM and PPM source images in human-readable
ASCII form. The generator converts those sources with
ImageMagick because `cjxl` 0.12 cannot read their P2/P3 encoding directly.

## Regenerating

From the repository root, run:

```sh
scripts/generate-decode-fixtures.sh
```

The script overwrites every generated fixture, then prints its JPEG XL metadata
and SHA-256 checksum for review. The fixtures were last generated with:

- libjxl command-line tools 0.12.0 (`cjxl` and `jxlinfo`)
- ImageMagick 7.1.2-13 (`magick`)
- libjpeg-turbo 3.2.0, build 20260630 (`cjpeg`)
- exif 0.6.22
- Exiv2 0.28.6
- `/usr/share/color/icc/colord/AdobeRGB1998.icc`, SHA-256
  `5bcba9d65c9f106d6b4ad05f477a1457437b7718aa0c903f5f002df1d0f575b5`

Review the generated metadata and Git diff, then run `cargo test`.

## Fixture groups

| Files | Purpose |
| --- | --- |
| `test_pattern-{sRGB,HLG,PQ}.jxl` | Identical 16-bit test-pattern samples tagged as sRGB, HLG, and PQ. |
| `ramp-hlg-*.jxl`, `ramp-pq-*.jxl` | Linear-in-luminance HDR ramps with explicit or default intensity targets. |
| `grayscale.jxl`, `linear-bt2020.jxl`, `reference-{hlg,pq}.jxl` | Basic channel and enumerated color-space handling. |
| `oriented.jxl`, `alpha.jxl`, `associated-alpha.jxl`, `animation.jxl` | Orientation, alpha, and animation decoding. |
| `icc.jxl`, `exif-common.jxl`, `xmp-rating.jxl` | Embedded ICC, EXIF, and XMP metadata. |
