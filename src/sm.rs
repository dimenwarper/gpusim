/// Streaming Multiprocessor (SM) — the core compute unit of a GPU.
/// Each SM contains warp schedulers, tensor cores, CUDA cores, and fast shared memory (SMEM).
use crate::warp::{Warp, WarpScheduler};
use crate::tensor_core::TensorCore;

pub const SMEM_SIZE_BYTES: usize = 256 * 1024; // 256KB per SM
pub const WARPS_PER_SM: usize = 4;

pub struct StreamingMultiprocessor {
    pub id: usize,
    /// Fast on-chip shared memory (programmer-controlled cache)
    pub smem: Box<[u8; SMEM_SIZE_BYTES]>,
    /// Warp schedulers (one per subpartition)
    pub warp_schedulers: Vec<WarpScheduler>,
    /// Tensor cores for matrix multiplication (one per subpartition)
    pub tensor_cores: Vec<TensorCore>,
}

impl StreamingMultiprocessor {
    pub fn new(id: usize) -> Self {
        StreamingMultiprocessor {
            id,
            smem: Box::new([0u8; SMEM_SIZE_BYTES]),
            warp_schedulers: (0..WARPS_PER_SM).map(|_| WarpScheduler::new()).collect(),
            tensor_cores: (0..WARPS_PER_SM).map(|_| TensorCore::new()).collect(),
        }
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
