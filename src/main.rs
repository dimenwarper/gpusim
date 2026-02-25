use gpusim::gpu::GPU;
use gpusim::kernel::{Dim3, Kernel, LaunchConfig};
use gpusim::scheduler::SchedulingPolicy;

fn main() {
    let mut gpu = GPU::h100();
    println!(
        "Initialized H100-like GPU with {} SMs, {}MB L2 cache, {}GB HBM",
        gpu.sms.len(),
        gpu.l2_cache.size_bytes / (1024 * 1024),
        gpu.hbm.size_bytes / (1024 * 1024 * 1024),
    );

    // --- Vector addition demo ---
    // C[i] = A[i] + B[i] for N elements, each stored as a single f32 (4 bytes)
    let n: u32 = 1024;
    let stride = std::mem::size_of::<f32>();

    // Write input arrays A and B into HBM
    // A starts at address 0, B at n*4, C at 2*n*4
    let base_a: usize = 0;
    let base_b: usize = n as usize * stride;
    let base_c: usize = 2 * n as usize * stride;

    for i in 0..n as usize {
        let a_val: f32 = i as f32;
        let b_val: f32 = (n as usize - i) as f32;
        gpu.hbm.write(base_a + i * stride, &a_val.to_le_bytes());
        gpu.hbm.write(base_b + i * stride, &b_val.to_le_bytes());
    }

    // Define the vector addition kernel
    let kernel = Kernel::new("vec_add", move |ctx| {
        let i = ctx.global_id() as usize;
        if i >= n as usize {
            return;
        }
        let addr_a = base_a + i * stride;
        let addr_b = base_b + i * stride;
        let addr_c = base_c + i * stride;

        let a = f32::from_le_bytes(ctx.gmem.read(addr_a, 4).try_into().unwrap());
        let b = f32::from_le_bytes(ctx.gmem.read(addr_b, 4).try_into().unwrap());
        let c = a + b;
        ctx.gmem.write(addr_c, &c.to_le_bytes());
    });

    // Launch: 1 thread per element, 128 threads per block
    // Using 32 registers/thread so the occupancy calculator has something to work with
    let threads_per_block = 128u32;
    let num_blocks = n.div_ceil(threads_per_block);
    let config = LaunchConfig::new(Dim3::x(num_blocks), Dim3::x(threads_per_block))
        .with_resources(32, 0); // 32 regs/thread, no explicit SMEM

    let stats = gpu.launch_kernel(&kernel, &config, SchedulingPolicy::Gto);

    // Verify results
    let mut all_correct = true;
    for i in 0..n as usize {
        let bytes = gpu.hbm.read(base_c + i * stride, 4);
        let c = f32::from_le_bytes(bytes.try_into().unwrap());
        let expected = n as f32; // a[i] + b[i] = i + (n - i) = n
        if (c - expected).abs() > 1e-5 {
            println!("MISMATCH at i={}: got {}, expected {}", i, c, expected);
            all_correct = false;
        }
    }

    if all_correct {
        println!("Verification PASSED: all {} results correct (each = {})", n, n);
    }

    println!(
        "Stats: {} blocks | {} warps | {} threads | occupancy={:.1}% (limited by {}) | policy={}",
        stats.blocks_executed,
        stats.warps_executed,
        stats.threads_executed,
        stats.theoretical_occupancy * 100.0,
        stats.occupancy_limiter,
        stats.scheduling_policy,
    );
}
