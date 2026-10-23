//! Integer conversions and byte and time units shared by the library and the
//! application.

use std::time::Duration;

pub const BYTES_PER_MIB: u64 = 1024 * 1024;

/// Converts geometry known to fit, without a fallible-looking error path.
///
/// # Panics
///
/// Panics on targets with pointers narrower than 32 bits, which are unsupported.
#[must_use]
pub fn usize_from_u32(value: u32) -> usize {
    usize::try_from(value).expect("u32 geometry fits usize on 32-bit-or-wider targets")
}

/// Converts byte counts known to fit, without a fallible-looking error path.
///
/// # Panics
///
/// Panics on targets with pointers wider than 64 bits, which are unsupported.
#[must_use]
pub fn u64_from_usize(value: usize) -> u64 {
    u64::try_from(value).expect("usize byte counts fit u64 on targets with at most 64-bit pointers")
}

#[must_use]
pub fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[must_use]
pub const fn mib_to_bytes(mib: u64) -> u64 {
    mib.saturating_mul(BYTES_PER_MIB)
}

// A u32 KiB count converts to f64 losslessly without a cast. Values
// saturate at 4 TiB, far above any diagnostic quantity.
#[must_use]
pub fn bytes_to_mib(bytes: u64) -> f64 {
    let kib = u32::try_from(bytes / 1024).unwrap_or(u32::MAX);
    f64::from(kib) / 1024.0
}

#[must_use]
pub fn format_mib(bytes: u64) -> String {
    let formatted = format!("{:.1}", bytes_to_mib(bytes));
    formatted
        .strip_suffix(".0")
        .unwrap_or(&formatted)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mib_format_omits_only_a_zero_fraction() {
        assert_eq!(format_mib(0), "0");
        assert_eq!(format_mib(1024 * 1024), "1");
        assert_eq!(format_mib(3 * 1024 * 1024 / 2), "1.5");
    }
}
