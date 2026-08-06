//! Text reports for GPU startup, image resources, and completed presentation.

use std::fmt::Write as _;

use wgpu::{SurfaceColorSpace, SurfaceColorSpaces, TextureFormat};

use super::hdr_metadata::{HdrSurface, SignalStatus};
use super::output::SurfaceCandidate;
use crate::cli::OutputMode;
use crate::units::{bytes_to_mib, u64_from_usize};
use xl_view::decode::SourceDynamicRange;

const EXPLICIT_COLOR_SPACES: [SurfaceColorSpace; 7] = [
    SurfaceColorSpace::Srgb,
    SurfaceColorSpace::ExtendedSrgbLinear,
    SurfaceColorSpace::DisplayP3,
    SurfaceColorSpace::Bt2100Pq,
    SurfaceColorSpace::Bt2100Hlg,
    SurfaceColorSpace::ExtendedSrgb,
    SurfaceColorSpace::ExtendedDisplayP3,
];

const HLG_PATTERN_OVERVIEW: &str = r"ITU-R BT.2111-3-derived HLG source pattern
source: narrow-range HLG/BT.2020, nominal peak 1,000 nits

+-----------+---------+----------+---------+---------+---------+----------+---------+-----------+
|           |  100 W  |  100 Y   |  100 C  |  100 G  |  100 M  |  100 R   |  100 B  |           |
|           +---------+----------+---------+---------+---------+----------+---------+           |
|           |         |          |         |         |         |          |         |           |
|           |         |          |         |         |         |          |         |           |
|           |         |          |         |         |         |          |         |           |
|           |         |          |         |         |         |          |         |           |
|           |         |          |         |         |         |          |         |           |
| 40% grey  |  75% W  |  75% Y   |  75% C  |  75% G  |  75% M  |  75% R   |  75% B  | 40% grey  |
|           |         |          |         |         |         |          |         |           |
|           |         |          |         |         |         |          |         |           |
|           |         |          |         |         |         |          |         |           |
|           |         |          |         |         |         |          |         |           |
|           |         |          |         |         |         |          |         |           |
|           |         |          |         |         |         |          |         |           |
|           |         |          |         |         |         |          |         |           |
+-----------+---------+----+-----+----+----+----+----+----+----+-----+----+----+----+-----------+
| 75% white |   -7%   | 0  | 10  | 20 | 30 | 40 | 50 | 60 | 70 | 80  | 90 |100 |109 | 75% white |
+-----------+---------+----+-----+----+-+--+----+----+----+----+-----+----+----+----+------+----+
| 0% black  |            -7%            |             ramp through 0% and 100%             |109%|
+---+---+---+------+--+---+--+---+--+---+-------+---------------------+-------------+---+--++---+
|   |   |   |      |  |   |  |   |  |           |                     |             |   |   |   |
|   |   |   |      |  |   |  |   |  |           |                     |             |   |   |   |
| Y | C | G |0% blk|-2| 0 |+2| 0 |+4| 0% black  |      75% white      |  0% black   | M | R | B |
|   |   |   |      |  |   |  |   |  |           |                     |             |   |   |   |
|709|709|709|      |  |   |  |   |  |           |                     |             |709|709|709|
|   |   |   |      |  |   |  |   |  |           |                     |             |   |   |   |
+---+---+---+------+--+---+--+---+--+-----------+---------------------+-------------+---+---+---+

W/Y/C/G/M/R/B = white/yellow/cyan/green/magenta/red/blue
709             = BT.709-equivalent colour bars";

pub(super) struct ImageWorkDiagnostics {
    pub(super) tile_cache: String,
    pub(super) tile_hits: u64,
    pub(super) tile_misses: u64,
    pub(super) viewport_resampler: String,
    pub(super) resampling_scratch_peak_bytes: u64,
    pub(super) gpu_image_budget_bytes: u64,
    pub(super) gpu_memory_limit_bytes: u64,
}

pub(super) struct ImageDiagnostics {
    pub(super) background: &'static str,
    pub(super) exposure_stops: f32,
    pub(super) hdr_metadata_status: SignalStatus,
    pub(super) cpu_storage_bytes: usize,
    pub(super) upload_staging_buffer_bytes: usize,
    pub(super) allocator_report: Option<wgpu::AllocatorReport>,
    pub(super) coarse_mip_levels: u32,
    pub(super) work: ImageWorkDiagnostics,
}

pub(super) fn startup_report(
    adapter: &wgpu::Adapter,
    surface: &HdrSurface<'_>,
    output_mode: OutputMode,
    source_dynamic_range: SourceDynamicRange,
    selected: SurfaceCandidate,
    hdr_metadata_status: SignalStatus,
    diagnostics_pattern: bool,
) -> String {
    let mut report = surface_capabilities_report(
        adapter,
        surface,
        output_mode,
        source_dynamic_range,
        selected,
    );
    #[cfg(not(target_vendor = "apple"))]
    writeln!(
        report,
        "Vulkan HDR metadata extension: {}",
        if surface.is_metadata_supported() {
            "enabled"
        } else {
            "unavailable"
        }
    )
    .unwrap();
    writeln!(report, "HDR metadata state: {hdr_metadata_status}").unwrap();
    if diagnostics_pattern {
        writeln!(report).unwrap();
        report.push_str(HLG_PATTERN_OVERVIEW);
    }
    report
}

pub(super) fn image_report(diagnostics: ImageDiagnostics) -> String {
    let ImageDiagnostics {
        background,
        exposure_stops,
        hdr_metadata_status,
        cpu_storage_bytes,
        upload_staging_buffer_bytes,
        allocator_report,
        coarse_mip_levels,
        work,
    } = diagnostics;
    let cpu_storage_mib = bytes_to_mib(u64_from_usize(cpu_storage_bytes));
    let upload_staging_buffer_mib = bytes_to_mib(u64_from_usize(upload_staging_buffer_bytes));
    let allocator_memory = allocator_memory_status(allocator_report);
    let mut report = String::new();
    writeln!(report, "  background: {background}").unwrap();
    writeln!(report, "  exposure: {exposure_stops:+.2} stops").unwrap();
    writeln!(report, "  HDR metadata state: {hdr_metadata_status}").unwrap();
    writeln!(
        report,
        "  decoded CPU storage: {cpu_storage_mib:.2} MiB (full-resolution RGBA16F + coarse RGBA32F preview)"
    )
    .unwrap();
    writeln!(
        report,
        "  temporary CPU-to-GPU upload buffer: up to {upload_staging_buffer_mib:.2} MiB"
    )
    .unwrap();
    writeln!(
        report,
        "  GPU image budget estimate: {:.2} MiB of {:.2} MiB",
        bytes_to_mib(work.gpu_image_budget_bytes),
        bytes_to_mib(work.gpu_memory_limit_bytes),
    )
    .unwrap();
    writeln!(report, "  GPU allocator memory: {allocator_memory}").unwrap();
    writeln!(report, "  coarse-preview mip levels: {coarse_mip_levels}").unwrap();
    write_image_work_status(&mut report, &work, false);
    report
}

pub(super) fn image_finished_report(diagnostics: &ImageWorkDiagnostics) -> String {
    let mut report = String::new();
    write_image_work_status(&mut report, diagnostics, true);
    write!(
        report,
        "  GPU image budget estimate: {:.2} MiB of {:.2} MiB",
        bytes_to_mib(diagnostics.gpu_image_budget_bytes),
        bytes_to_mib(diagnostics.gpu_memory_limit_bytes),
    )
    .unwrap();
    report
}

fn write_image_work_status(
    report: &mut String,
    diagnostics: &ImageWorkDiagnostics,
    trailing_newline: bool,
) {
    writeln!(
        report,
        "  tile cache: {} (hits {}, misses {})",
        diagnostics.tile_cache, diagnostics.tile_hits, diagnostics.tile_misses,
    )
    .unwrap();
    writeln!(
        report,
        "  viewport resampling: {}",
        diagnostics.viewport_resampler
    )
    .unwrap();
    let scratch_mib = bytes_to_mib(diagnostics.resampling_scratch_peak_bytes);
    if trailing_newline {
        writeln!(
            report,
            "  CPU resampling scratch peak: {scratch_mib:.2} MiB"
        )
        .unwrap();
    } else {
        write!(
            report,
            "  CPU resampling scratch peak: {scratch_mib:.2} MiB"
        )
        .unwrap();
    }
}

fn surface_capabilities_report(
    adapter: &wgpu::Adapter,
    surface: &HdrSurface<'_>,
    output_mode: OutputMode,
    source_dynamic_range: SourceDynamicRange,
    selected: SurfaceCandidate,
) -> String {
    let info = adapter.get_info();
    let capabilities = surface.get_capabilities(adapter);
    let hdr_info = surface.display_hdr_info(adapter);
    let mut report = String::new();

    writeln!(report, "GPU adapter:").unwrap();
    writeln!(report, "  name: {}", info.name).unwrap();
    writeln!(report, "  type: {:?}", info.device_type).unwrap();
    writeln!(report, "  vendor: {:#06x}", info.vendor).unwrap();
    writeln!(report, "  device: {:#06x}", info.device).unwrap();
    writeln!(report, "  driver: {} ({})", info.driver, info.driver_info).unwrap();
    writeln!(report, "  backend: {:?}", info.backend).unwrap();
    writeln!(report).unwrap();
    writeln!(report, "Advertised surface format/color-space pairs:").unwrap();

    let mut pair_count = 0;
    for format_capability in &capabilities.format_capabilities {
        for color_space in EXPLICIT_COLOR_SPACES {
            let flag = color_space
                .to_color_spaces()
                .expect("the explicit color-space list must not contain Auto");

            if format_capability.color_spaces.contains(flag) {
                writeln!(
                    report,
                    "  {:?} + {:?}",
                    format_capability.format, color_space
                )
                .unwrap();
                pair_count += 1;
            }
        }

        write_unknown_color_space_bits(
            &mut report,
            format_capability.format,
            format_capability.color_spaces,
        );
    }

    if pair_count == 0 {
        writeln!(report, "  (none)").unwrap();
    }

    writeln!(report).unwrap();
    writeln!(report, "Requested output mode: {}", output_mode.as_str()).unwrap();
    if output_mode == OutputMode::Auto {
        writeln!(report, "Automatic source class: {source_dynamic_range:?}").unwrap();
    }
    writeln!(
        report,
        "Selected surface pair: {:?} + {:?}",
        selected.format, selected.color_space
    )
    .unwrap();
    writeln!(report, "Present modes: {:?}", capabilities.present_modes).unwrap();
    writeln!(report, "Alpha modes: {:?}", capabilities.alpha_modes).unwrap();
    writeln!(report, "Surface usages: {:?}", capabilities.usages).unwrap();
    writeln!(report, "Display HDR information: {hdr_info:#?}").unwrap();
    report
}

fn write_unknown_color_space_bits(
    report: &mut String,
    format: TextureFormat,
    advertised: SurfaceColorSpaces,
) {
    let known = EXPLICIT_COLOR_SPACES
        .into_iter()
        .filter_map(SurfaceColorSpace::to_color_spaces)
        .fold(SurfaceColorSpaces::empty(), |all, flag| all | flag);
    let unknown = advertised.difference(known);

    if !unknown.is_empty() {
        writeln!(report, "  {format:?} + unknown bits {unknown:?}").unwrap();
    }
}

fn allocator_memory_status(report: Option<wgpu::AllocatorReport>) -> String {
    report.map_or_else(
        || "unavailable".to_owned(),
        |report| {
            format!(
                "{:.2} MiB live allocations, {:.2} MiB reserved blocks",
                bytes_to_mib(report.total_allocated_bytes),
                bytes_to_mib(report.total_reserved_bytes),
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn allocator_diagnostics_distinguish_live_allocations_from_reserved_blocks() {
        let report = wgpu::AllocatorReport {
            allocations: Vec::new(),
            blocks: Vec::new(),
            total_allocated_bytes: 96 * MIB,
            total_reserved_bytes: 160 * MIB,
        };

        assert_eq!(
            allocator_memory_status(Some(report)),
            "96.00 MiB live allocations, 160.00 MiB reserved blocks"
        );
        assert_eq!(allocator_memory_status(None), "unavailable");
    }

    #[test]
    fn image_reports_preserve_their_line_and_newline_contracts() {
        let report = image_report(ImageDiagnostics {
            background: "black",
            exposure_stops: 1.25,
            hdr_metadata_status: SignalStatus::NotRequested,
            cpu_storage_bytes: usize::try_from(2 * MIB).unwrap(),
            upload_staging_buffer_bytes: usize::try_from(MIB / 2).unwrap(),
            allocator_report: None,
            coarse_mip_levels: 4,
            work: image_work_diagnostics(),
        });

        assert_eq!(
            report,
            concat!(
                "  background: black\n",
                "  exposure: +1.25 stops\n",
                "  HDR metadata state: HDR metadata not requested\n",
                "  decoded CPU storage: 2.00 MiB ",
                "(full-resolution RGBA16F + coarse RGBA32F preview)\n",
                "  temporary CPU-to-GPU upload buffer: up to 0.50 MiB\n",
                "  GPU image budget estimate: 1.50 MiB of 4.00 MiB\n",
                "  GPU allocator memory: unavailable\n",
                "  coarse-preview mip levels: 4\n",
                "  tile cache: 2 of 3 working-set slots resident (hits 8, misses 5)\n",
                "  viewport resampling: active 800x600\n",
                "  CPU resampling scratch peak: 0.25 MiB",
            )
        );

        assert_eq!(
            image_finished_report(&image_work_diagnostics()),
            concat!(
                "  tile cache: 2 of 3 working-set slots resident (hits 8, misses 5)\n",
                "  viewport resampling: active 800x600\n",
                "  CPU resampling scratch peak: 0.25 MiB\n",
                "  GPU image budget estimate: 1.50 MiB of 4.00 MiB",
            )
        );
    }

    fn image_work_diagnostics() -> ImageWorkDiagnostics {
        ImageWorkDiagnostics {
            tile_cache: "2 of 3 working-set slots resident".to_owned(),
            tile_hits: 8,
            tile_misses: 5,
            viewport_resampler: "active 800x600".to_owned(),
            resampling_scratch_peak_bytes: MIB / 4,
            gpu_image_budget_bytes: MIB / 2 * 3,
            gpu_memory_limit_bytes: 4 * MIB,
        }
    }
}
