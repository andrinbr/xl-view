pub(crate) fn usize_from_u32(value: u32) -> usize {
    usize::try_from(value).expect("u32 geometry fits usize on 32-bit-or-wider targets")
}

pub(crate) fn u64_from_usize(value: usize) -> u64 {
    u64::try_from(value).expect("usize byte counts fit u64 on targets with at most 64-bit pointers")
}

pub(crate) fn bytes_to_mib(bytes: u64) -> f64 {
    let kib = u32::try_from(bytes / 1024).unwrap_or(u32::MAX);
    f64::from(kib) / 1024.0
}

pub(crate) fn format_mib(bytes: u64) -> String {
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
