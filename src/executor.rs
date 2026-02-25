/// Kernel execution engine.
/// Distributes thread blocks across SMs (round-robin), groups threads into
/// warps, and drives execution of each thread through the kernel function.
use crate::gpu::GPU;
use crate::kernel::{Dim3, Kernel, LaunchConfig, ThreadCtx};
use crate::warp::{Warp, WARP_SIZE};

/// Statistics collected during a kernel launch.
#[derive(Debug, Default)]
pub struct ExecutionStats {
    /// Total number of thread blocks executed
    pub blocks_executed: u32,
    /// Total number of warps executed
    pub warps_executed: u32,
    /// Total number of threads executed
    pub threads_executed: u32,
}

/// Executes a kernel on a GPU, simulating the SM/warp/thread hierarchy.
pub struct KernelExecutor<'a> {
    pub gpu: &'a mut GPU,
}

impl<'a> KernelExecutor<'a> {
    pub fn new(gpu: &'a mut GPU) -> Self {
        KernelExecutor { gpu }
    }

    /// Launch a kernel with the given configuration.
    /// Blocks are assigned to SMs round-robin; threads within each block
    /// are grouped into warps of 32 and executed sequentially (simulated SIMD).
    pub fn launch(&mut self, kernel: &Kernel, config: &LaunchConfig) -> ExecutionStats {
        let mut stats = ExecutionStats::default();
        let num_sms = self.gpu.sms.len();

        println!(
            "[gpusim] Launching kernel '{}': grid=({},{},{}) block=({},{},{})",
            kernel.name,
            config.grid_dim.x, config.grid_dim.y, config.grid_dim.z,
            config.block_dim.x, config.block_dim.y, config.block_dim.z,
        );

        // Iterate over all blocks in the grid
        for bz in 0..config.grid_dim.z {
            for by in 0..config.grid_dim.y {
                for bx in 0..config.grid_dim.x {
                    let block_idx = Dim3::new(bx, by, bz);

                    // Assign block to an SM (round-robin by flat block index)
                    let flat_block_id = (bz * config.grid_dim.y * config.grid_dim.x
                        + by * config.grid_dim.x
                        + bx) as usize;
                    let sm_id = flat_block_id % num_sms;

                    // Allocate per-block shared memory (SMEM)
                    let smem_size = self.gpu.sms[sm_id].smem.len();
                    let mut smem: Vec<u8> = vec![0u8; smem_size];

                    // Execute all threads in this block, grouped into warps
                    self.execute_block(
                        kernel,
                        config,
                        block_idx,
                        &mut smem,
                        &mut stats,
                    );

                    stats.blocks_executed += 1;
                }
            }
        }

        println!(
            "[gpusim] Kernel '{}' complete: {} blocks, {} warps, {} threads",
            kernel.name,
            stats.blocks_executed,
            stats.warps_executed,
            stats.threads_executed,
        );

        stats
    }

    /// Execute all threads in a single thread block, grouped into warps.
    fn execute_block(
        &mut self,
        kernel: &Kernel,
        config: &LaunchConfig,
        block_idx: Dim3,
        smem: &mut Vec<u8>,
        stats: &mut ExecutionStats,
    ) {
        let threads_per_block = config.threads_per_block() as usize;
        let warp_size = WARP_SIZE;

        // Divide block threads into warps (last warp may be partial)
        let num_warps = threads_per_block.div_ceil(warp_size);

        for warp_id in 0..num_warps {
            let warp_start = warp_id * warp_size;
            let warp_end = (warp_start + warp_size).min(threads_per_block);

            // Each lane in the warp corresponds to one thread
            for lane in warp_start..warp_end {
                // Map flat thread index back to Dim3
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

            // Track warp execution (simulates warp scheduler tick)
            let _ = Warp::new(warp_id);
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
