//! System memory information shared by decode and application budget policies.

use sysinfo::System;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryQueryError {
    #[error("failed to initialize the system memory query: {0}")]
    Initialization(#[from] sysinfo::Error),

    #[error("system reported zero total memory")]
    Unavailable,
}

/// Returns total system RAM in bytes.
///
/// # Errors
///
/// Returns an error if sysinfo cannot initialize or reports zero total memory.
pub fn total_system_memory_bytes() -> Result<u64, MemoryQueryError> {
    let mut system = System::new()?;
    system.refresh_memory();
    validate_total_memory_bytes(system.total_memory())
}

fn validate_total_memory_bytes(total: u64) -> Result<u64, MemoryQueryError> {
    if total == 0 {
        return Err(MemoryQueryError::Unavailable);
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_total_memory_is_unavailable() {
        assert!(matches!(
            validate_total_memory_bytes(0),
            Err(MemoryQueryError::Unavailable)
        ));
    }

    #[test]
    fn nonzero_total_memory_is_preserved() {
        for total in [1, 8 * 1024 * 1024 * 1024, u64::MAX] {
            assert_eq!(validate_total_memory_bytes(total).unwrap(), total);
        }
    }
}
