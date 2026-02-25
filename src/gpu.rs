/// Top-level GPU simulator.
/// Models a GPU as a collection of Streaming Multiprocessors (SMs)
/// connected to a shared memory hierarchy (L2 cache + HBM).
use crate::executor::{ExecutionStats, KernelExecutor};
use crate::kernel::{Kernel, LaunchConfig};
use crate::memory::{HBM, L2Cache};
use crate::occupancy::SmConfig;
use crate::scheduler::SchedulingPolicy;
use crate::sm::StreamingMultiprocessor;

pub struct GPU {
    /// All SMs on the GPU
    pub sms: Vec<StreamingMultiprocessor>,
    /// Shared L2 cache across all SMs
    pub l2_cache: L2Cache,
    /// High Bandwidth Memory (main GPU memory)
    pub hbm: HBM,
    /// SM hardware configuration (used for occupancy calculations)
    pub sm_config: SmConfig,
}

impl GPU {
    pub fn new(
        num_sms: usize,
        l2_size_bytes: usize,
        hbm_size_bytes: usize,
        sm_config: SmConfig,
    ) -> Self {
        let sms = (0..num_sms)
            .map(|id| StreamingMultiprocessor::new(id))
            .collect();

        GPU {
            sms,
            l2_cache: L2Cache::new(l2_size_bytes),
            hbm: HBM::new(hbm_size_bytes),
            sm_config,
        }
    }

    /// Create an H100-like GPU configuration.
    pub fn h100() -> Self {
        Self::new(
            132,                     // 132 SMs
            50 * 1024 * 1024,        // ~50MB L2 cache
            80 * 1024 * 1024 * 1024, // 80GB HBM
            SmConfig::h100(),
        )
    }

    /// Launch a kernel with the given scheduling policy.
    pub fn launch_kernel(
        &mut self,
        kernel: &Kernel,
        config: &LaunchConfig,
        policy: SchedulingPolicy,
    ) -> ExecutionStats {
        let sm_config = self.sm_config.clone();
        let mut executor = KernelExecutor::new(self, policy, sm_config);
        executor.launch(kernel, config)
    }
}
