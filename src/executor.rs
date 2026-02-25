/// Kernel execution engine.
///
/// Implements two levels of scheduling:
///
/// 1. Block scheduling (GigaThread Engine equivalent):
///    Assigns thread blocks to SMs based on resource availability — the SM
///    with the most remaining headroom (vs. its occupancy limit) gets the
///    next block. Ties broken by SM ID (effectively round-robin among equals).
///
/// 2. Warp scheduling (per SM):
///    Within each block, warps are ordered by the chosen policy (LRR, GTO,
///    or TwoLevel) and executed in that order.
use crate::gpu::GPU;
use crate::kernel::{Dim3, Kernel, LaunchConfig, ThreadCtx};
use crate::occupancy::{max_blocks_per_sm, occupancy, KernelResources, SmConfig};
use crate::scheduler::{SchedulingPolicy, WarpScheduler, WarpSlot};
use crate::warp::WARP_SIZE;

/// Statistics collected during a kernel launch.
#[derive(Debug, Default)]
pub struct ExecutionStats {
    /// Total thread blocks executed
    pub blocks_executed: u32,
    /// Total warps executed
    pub warps_executed: u32,
    /// Total threads executed
    pub threads_executed: u32,
    /// Theoretical occupancy [0.0, 1.0]
    pub theoretical_occupancy: f32,
    /// Max blocks allowed per SM by occupancy constraints
    pub max_blocks_per_sm: u32,
    /// Which resource limited occupancy
    pub occupancy_limiter: String,
    /// Name of the warp scheduling policy used
    pub scheduling_policy: String,
}

/// Executes a kernel on a GPU, simulating the SM/warp/thread hierarchy.
pub struct KernelExecutor<'a> {
    pub gpu: &'a mut GPU,
    scheduler: Box<dyn WarpScheduler>,
    sm_config: SmConfig,
    /// Monotonically increasing counter for assigning warp ages
    warp_age_counter: u64,
}

impl<'a> KernelExecutor<'a> {
    pub fn new(gpu: &'a mut GPU, policy: SchedulingPolicy, sm_config: SmConfig) -> Self {
        KernelExecutor {
            scheduler: policy.build(),
            gpu,
            sm_config,
            warp_age_counter: 0,
        }
    }

    /// Launch a kernel with the given configuration.
    pub fn launch(&mut self, kernel: &Kernel, config: &LaunchConfig) -> ExecutionStats {
        let mut stats = ExecutionStats {
            scheduling_policy: self.scheduler.name().to_string(),
            ..Default::default()
        };

        // Build kernel resource profile for occupancy calculation
        let kernel_res = KernelResources {
            threads_per_block: config.threads_per_block(),
            regs_per_thread: config.regs_per_thread,
            smem_per_block: config.smem_per_block,
        };

        let (max_blks, limiter) = max_blocks_per_sm(&kernel_res, &self.sm_config);
        let warps_per_block = config.threads_per_block().div_ceil(32);
        let occ = occupancy(max_blks, warps_per_block, self.sm_config.max_warps);

        stats.max_blocks_per_sm = max_blks;
        stats.theoretical_occupancy = occ;
        stats.occupancy_limiter = limiter.to_string();

        println!(
            "[gpusim] Launching kernel '{}' | grid=({},{},{}) block=({},{},{}) | \
             policy={} | max_blocks/SM={} | occupancy={:.1}% (limited by {})",
            kernel.name,
            config.grid_dim.x, config.grid_dim.y, config.grid_dim.z,
            config.block_dim.x, config.block_dim.y, config.block_dim.z,
            stats.scheduling_policy,
            max_blks,
            occ * 100.0,
            limiter,
        );

        // Reset SM resource usage before launch
        for sm in self.gpu.sms.iter_mut() {
            sm.resource_usage = Default::default();
        }

        // Iterate over all blocks in the grid and assign them to SMs
        for bz in 0..config.grid_dim.z {
            for by in 0..config.grid_dim.y {
                for bx in 0..config.grid_dim.x {
                    let block_idx = Dim3::new(bx, by, bz);

                    // Find the SM with the most available headroom
                    let sm_id = self.find_best_sm(max_blks);

                    // Allocate resources on that SM
                    let warps = warps_per_block;
                    let smem = config.smem_per_block;
                    self.gpu.sms[sm_id].allocate_block(
                        config.threads_per_block(),
                        warps,
                        smem,
                    );

                    // Execute the block
                    let mut smem_buf = vec![0u8; config.smem_per_block.max(1) as usize];
                    self.execute_block(kernel, config, block_idx, &mut smem_buf, &mut stats);

                    // Free resources after block completes
                    self.gpu.sms[sm_id].free_block(
                        config.threads_per_block(),
                        warps,
                        smem,
                    );

                    stats.blocks_executed += 1;
                }
            }
        }

        println!(
            "[gpusim] Kernel '{}' complete | {} blocks | {} warps | {} threads | \
             occupancy={:.1}%",
            kernel.name,
            stats.blocks_executed,
            stats.warps_executed,
            stats.threads_executed,
            stats.theoretical_occupancy * 100.0,
        );

        stats
    }

    /// Find the SM with the most remaining block headroom (resource-availability-based
    /// scheduling, matching empirical NVIDIA GigaThread Engine behaviour).
    /// Ties broken by SM ID (lowest first).
    fn find_best_sm(&self, max_blocks: u32) -> usize {
        self.gpu
            .sms
            .iter()
            .enumerate()
            .filter(|(_, sm)| sm.resource_usage.active_blocks < max_blocks)
            .max_by_key(|(id, sm)| {
                let headroom = max_blocks.saturating_sub(sm.resource_usage.active_blocks);
                // Primary: headroom (higher = better); secondary: lower SM ID wins ties
                (headroom, usize::MAX - id)
            })
            .map(|(id, _)| id)
            // Fallback: SM 0 (should never happen with a valid grid)
            .unwrap_or(0)
    }

    /// Execute all threads in a single thread block, using the warp scheduler
    /// to determine warp execution order.
    fn execute_block(
        &mut self,
        kernel: &Kernel,
        config: &LaunchConfig,
        block_idx: Dim3,
        smem: &mut Vec<u8>,
        stats: &mut ExecutionStats,
    ) {
        let threads_per_block = config.threads_per_block() as usize;
        let num_warps = threads_per_block.div_ceil(WARP_SIZE);

        // Create warp slots for the scheduler, assigning ages in order
        let warp_slots: Vec<WarpSlot> = (0..num_warps)
            .map(|i| {
                let age = self.warp_age_counter + i as u64;
                WarpSlot::new(i, age)
            })
            .collect();
        self.warp_age_counter += num_warps as u64;

        // Get execution order from the warp scheduler
        let ordered = self.scheduler.order_warps(&warp_slots);

        for warp_idx in ordered {
            let warp_start = warp_idx * WARP_SIZE;
            let warp_end = (warp_start + WARP_SIZE).min(threads_per_block);

            // Execute all 32 lanes of the warp (simulated SIMD)
            for lane in warp_start..warp_end {
                let thread_idx = flat_to_dim3(lane as u32, config.block_dim);
                let mut ctx = ThreadCtx {
                    thread_idx,
                    block_idx,
                    block_dim: config.block_dim,
                    grid_dim: config.grid_dim,
                    smem,
                    gmem: &mut self.gpu.hbm,
                };
                (kernel.func)(&mut ctx);
                stats.threads_executed += 1;
            }

            self.scheduler.record_issued(warp_idx);
            stats.warps_executed += 1;
        }
    }
}

/// Convert a flat thread index into a Dim3 given block dimensions.
fn flat_to_dim3(flat: u32, block_dim: Dim3) -> Dim3 {
    let x = flat % block_dim.x;
    let y = (flat / block_dim.x) % block_dim.y;
    let z = flat / (block_dim.x * block_dim.y);
    Dim3::new(x, y, z)
}
