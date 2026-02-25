/// Memory hierarchy simulation.
/// Models the three tiers of GPU memory:
///   - SMEM: per-SM on-chip shared memory (in sm.rs)
///   - L2Cache: shared across all SMs
///   - HBM: main high-bandwidth memory

use std::collections::HashMap;

/// Shared L2 cache across all SMs (~50MB on H100).
/// Slower than SMEM but shared across the entire GPU.
/// Uses a sparse map to avoid eagerly allocating the full capacity.
pub struct L2Cache {
    pub size_bytes: usize,
    data: HashMap<usize, u8>,
}

impl L2Cache {
    pub fn new(size_bytes: usize) -> Self {
        L2Cache {
            size_bytes,
            data: HashMap::new(),
        }
    }

    pub fn read(&self, addr: usize, len: usize) -> Vec<u8> {
        (addr..addr + len)
            .map(|a| *self.data.get(&a).unwrap_or(&0))
            .collect()
    }

    pub fn write(&mut self, addr: usize, bytes: &[u8]) {
        for (i, &byte) in bytes.iter().enumerate() {
            self.data.insert(addr + i, byte);
        }
    }
}

/// High Bandwidth Memory — the main GPU memory (e.g., 80GB on H100, 3.4 TB/s bandwidth).
/// Slowest in the hierarchy but largest capacity.
/// Uses a sparse map to simulate large address spaces without real allocation.
pub struct HBM {
    pub size_bytes: usize,
    data: HashMap<usize, u8>,
    /// Simulated bandwidth in bytes per second
    pub bandwidth_bps: u64,
}

impl HBM {
    pub fn new(size_bytes: usize) -> Self {
        HBM {
            size_bytes,
            data: HashMap::new(),
            bandwidth_bps: 3_400_000_000_000, // 3.4 TB/s (H100)
        }
    }

    pub fn read(&self, addr: usize, len: usize) -> Vec<u8> {
        (addr..addr + len)
            .map(|a| *self.data.get(&a).unwrap_or(&0))
            .collect()
    }

    pub fn write(&mut self, addr: usize, bytes: &[u8]) {
        for (i, &byte) in bytes.iter().enumerate() {
            self.data.insert(addr + i, byte);
        }
    }
}
