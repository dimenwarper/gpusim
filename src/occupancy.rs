/// Occupancy calculation — determines how many thread blocks can simultaneously
/// reside on an SM, given the kernel's resource requirements and SM hardware limits.
/// Based on GPGPU-Sim's max_cta() logic and NVIDIA architecture whitepapers.

/// Hardware resource limits for a specific SM architecture.
#[derive(Debug, Clone)]
pub struct SmConfig {
    /// Maximum concurrent threads per SM
    pub max_threads: u32,
    /// Maximum concurrent warps per SM
    pub max_warps: u32,
    /// Hard hardware cap on concurrent thread blocks per SM
    pub max_blocks: u32,
    /// Total 32-bit registers in the SM register file
    pub total_regs: u32,
    /// Register allocation granularity (per warp, in registers)
    pub reg_alloc_granularity: u32,
    /// Total shared memory (SMEM) per SM in bytes
    pub total_smem_bytes: u32,
    /// Shared memory allocation granularity in bytes
    pub smem_alloc_granularity: u32,
}

impl SmConfig {
    /// H100 (Hopper, CC 9.0) SM configuration.
    pub fn h100() -> Self {
        SmConfig {
            max_threads: 2048,
            max_warps: 64,
            max_blocks: 32,
            total_regs: 65536,
            reg_alloc_granularity: 256,
            total_smem_bytes: 228 * 1024, // 228 KB
            smem_alloc_granularity: 128,
        }
    }

    /// A100 (Ampere, CC 8.0) SM configuration.
    pub fn a100() -> Self {
        SmConfig {
            max_threads: 2048,
            max_warps: 64,
            max_blocks: 32,
            total_regs: 65536,
            reg_alloc_granularity: 256,
            total_smem_bytes: 164 * 1024, // 164 KB
            smem_alloc_granularity: 128,
        }
    }
}

/// Resource requirements declared by a kernel at launch time.
#[derive(Debug, Clone)]
pub struct KernelResources {
    /// Number of threads per block
    pub threads_per_block: u32,
    /// Registers used per thread (0 = untracked, no register pressure)
    pub regs_per_thread: u32,
    /// Shared memory bytes per block (0 = none)
    pub smem_per_block: u32,
}

/// Which resource is limiting occupancy.
#[derive(Debug, Clone, PartialEq)]
pub enum OccupancyLimiter {
    ThreadSlots,
    WarpSlots,
    RegisterFile,
    SharedMemory,
    HardwareBlockCap,
}

impl std::fmt::Display for OccupancyLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OccupancyLimiter::ThreadSlots    => write!(f, "thread slots"),
            OccupancyLimiter::WarpSlots      => write!(f, "warp slots"),
            OccupancyLimiter::RegisterFile   => write!(f, "register file"),
            OccupancyLimiter::SharedMemory   => write!(f, "shared memory"),
            OccupancyLimiter::HardwareBlockCap => write!(f, "hardware block cap"),
        }
    }
}

fn round_up(val: u32, granularity: u32) -> u32 {
    if granularity == 0 {
        return val;
    }
    ((val + granularity - 1) / granularity) * granularity
}

/// Compute the maximum number of thread blocks that can simultaneously reside
/// on a single SM, and identify which resource is the bottleneck.
/// This is the minimum across five independent resource constraints:
///   1. Thread slots
///   2. Warp slots
///   3. Register file
///   4. Shared memory
///   5. Hardware block cap
pub fn max_blocks_per_sm(kernel: &KernelResources, sm: &SmConfig) -> (u32, OccupancyLimiter) {
    let threads = kernel.threads_per_block.max(1);
    let warps_per_block = threads.div_ceil(32);

    // Limiter 1: thread slots
    let by_threads = sm.max_threads / threads;

    // Limiter 2: warp slots
    let by_warps = sm.max_warps / warps_per_block;

    // Limiter 3: register file
    let by_regs = if kernel.regs_per_thread == 0 {
        u32::MAX
    } else {
        let regs_per_warp = round_up(kernel.regs_per_thread * 32, sm.reg_alloc_granularity);
        let regs_per_block = regs_per_warp * warps_per_block;
        if regs_per_block == 0 {
            u32::MAX
        } else {
            sm.total_regs / regs_per_block
        }
    };

    // Limiter 4: shared memory
    let by_smem = if kernel.smem_per_block == 0 {
        u32::MAX
    } else {
        let smem_rounded = round_up(kernel.smem_per_block, sm.smem_alloc_granularity);
        sm.total_smem_bytes / smem_rounded
    };

    // Limiter 5: hardware block cap
    let by_hw = sm.max_blocks;

    let max = by_threads.min(by_warps).min(by_regs).min(by_smem).min(by_hw);

    // Identify the bottleneck
    let limiter = if max == by_hw {
        OccupancyLimiter::HardwareBlockCap
    } else if max == by_smem {
        OccupancyLimiter::SharedMemory
    } else if max == by_regs {
        OccupancyLimiter::RegisterFile
    } else if max == by_warps {
        OccupancyLimiter::WarpSlots
    } else {
        OccupancyLimiter::ThreadSlots
    };

    (max, limiter)
}

/// Theoretical occupancy as a fraction [0.0, 1.0].
/// occupancy = (resident warps) / (max warps per SM)
pub fn occupancy(max_blocks: u32, warps_per_block: u32, max_warps_per_sm: u32) -> f32 {
    let resident_warps = max_blocks * warps_per_block;
    resident_warps as f32 / max_warps_per_sm as f32
}
