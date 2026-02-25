/// Streaming Multiprocessor (SM) — the core compute unit of a GPU.
/// Each SM contains warp schedulers, tensor cores, CUDA cores, and fast shared memory (SMEM).
use crate::warp::{Warp, WarpScheduler};
use crate::tensor_core::TensorCore;

pub const SMEM_SIZE_BYTES: usize = 256 * 1024; // 256KB per SM
pub const WARPS_PER_SM: usize = 4;

/// Tracks live resource consumption on an SM, updated as blocks are assigned
/// and retired. Used by the block scheduler to compute available headroom.
#[derive(Debug, Default, Clone)]
pub struct SmResourceUsage {
    /// Number of thread blocks currently resident on this SM
    pub active_blocks: u32,
    /// Number of threads currently resident
    pub used_threads: u32,
    /// Number of warps currently resident
    pub used_warps: u32,
    /// Bytes of shared memory currently allocated
    pub used_smem_bytes: u32,
}

pub struct StreamingMultiprocessor {
    pub id: usize,
    /// Fast on-chip shared memory (programmer-controlled cache)
    pub smem: Box<[u8; SMEM_SIZE_BYTES]>,
    /// Warp schedulers (one per subpartition)
    pub warp_schedulers: Vec<WarpScheduler>,
    /// Tensor cores for matrix multiplication (one per subpartition)
    pub tensor_cores: Vec<TensorCore>,
    /// Live resource usage — updated by the executor as blocks are assigned/retired
    pub resource_usage: SmResourceUsage,
}

impl StreamingMultiprocessor {
    pub fn new(id: usize) -> Self {
        StreamingMultiprocessor {
            id,
            smem: Box::new([0u8; SMEM_SIZE_BYTES]),
            warp_schedulers: (0..WARPS_PER_SM).map(|_| WarpScheduler::new()).collect(),
            tensor_cores: (0..WARPS_PER_SM).map(|_| TensorCore::new()).collect(),
            resource_usage: SmResourceUsage::default(),
        }
    }

    /// Allocate resources for a new block. Returns false if the SM is full.
    pub fn allocate_block(&mut self, threads: u32, warps: u32, smem_bytes: u32) -> bool {
        self.resource_usage.active_blocks += 1;
        self.resource_usage.used_threads += threads;
        self.resource_usage.used_warps += warps;
        self.resource_usage.used_smem_bytes += smem_bytes;
        true
    }

    /// Free resources when a block retires.
    pub fn free_block(&mut self, threads: u32, warps: u32, smem_bytes: u32) {
        self.resource_usage.active_blocks = self.resource_usage.active_blocks.saturating_sub(1);
        self.resource_usage.used_threads = self.resource_usage.used_threads.saturating_sub(threads);
        self.resource_usage.used_warps = self.resource_usage.used_warps.saturating_sub(warps);
        self.resource_usage.used_smem_bytes = self.resource_usage.used_smem_bytes.saturating_sub(smem_bytes);
    }

    /// Dispatch a warp to an available warp scheduler
    pub fn dispatch_warp(&mut self, warp: Warp) -> bool {
        for scheduler in self.warp_schedulers.iter_mut() {
            if scheduler.is_idle() {
                scheduler.load_warp(warp);
                return true;
            }
        }
        false // No available scheduler
    }
}
