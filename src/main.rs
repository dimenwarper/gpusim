use gpusim::cluster::{Cluster, DeviceId};
use gpusim::gpu::GPU;
use gpusim::interconnect::AllReduceAlgorithm;
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

    // -----------------------------------------------------------------------
    // Multi-GPU cluster demo
    // -----------------------------------------------------------------------
    println!("\n{}", "=".repeat(60));
    println!("Multi-GPU Cluster Demo");
    println!("{}", "=".repeat(60));

    // 2 nodes × 8 H100s = 16 GPUs, NVLink 4.0 + NDR InfiniBand
    let mut cluster = Cluster::h100_dgx(2);
    println!(
        "Cluster: {} nodes × {} GPUs = {} GPUs total",
        cluster.nodes.len(),
        cluster.nodes[0].gpus.len(),
        cluster.total_gpus(),
    );
    println!(
        "         NVLink {:.0} GB/s intra-node | InfiniBand {:.0} GB/s inter-node\n",
        cluster.nodes[0].nvlink.bandwidth_gb_s,
        cluster.infiniband.bandwidth_gb_s,
    );

    // --- Point-to-point transfers ---
    println!("--- Point-to-Point Transfers (1 GB) ---");
    let one_gb = 1u64 << 30;

    let transfers = [
        (DeviceId::new(0, 0), DeviceId::new(0, 0), "same device"),
        (DeviceId::new(0, 0), DeviceId::new(0, 1), "intra-node (NVLink)"),
        (DeviceId::new(0, 0), DeviceId::new(1, 0), "inter-node (InfiniBand)"),
    ];

    for (src, dst, label) in &transfers {
        let t = cluster.transfer(*src, *dst, one_gb);
        if t.time_us == 0.0 {
            println!("  {:40}  instant", label);
        } else {
            println!(
                "  {:40}  {:8.2}ms  ({:.1} GB/s)",
                label,
                t.time_us / 1_000.0,
                t.effective_bandwidth_gb_s,
            );
        }
    }

    // --- AllReduce comparison ---
    println!("\n--- AllReduce: 1 GB/GPU across {} GPUs ---", cluster.total_gpus());
    println!("  {:<10} {:>12}  {:>14}  {:>12}", "Algorithm", "Time (ms)", "Bus BW (GB/s)", "Efficiency");
    println!("  {}", "-".repeat(55));

    for algo in [AllReduceAlgorithm::Ring, AllReduceAlgorithm::Tree, AllReduceAlgorithm::Direct] {
        let s = cluster.all_reduce(one_gb, algo);
        println!(
            "  {:<10} {:>12.2}  {:>14.1}  {:>11.1}%",
            s.algorithm,
            s.time_us / 1_000.0,
            s.bus_bandwidth_gb_s,
            s.efficiency * 100.0,
        );
    }

    // --- AllGather ---
    println!("\n--- AllGather: 128 MB/GPU across {} GPUs ---", cluster.total_gpus());
    let s = cluster.all_gather(128 * 1024 * 1024);
    println!(
        "  Ring:  {:.2}ms  ({:.1} GB/s bus bandwidth)",
        s.time_us / 1_000.0,
        s.bus_bandwidth_gb_s,
    );

    // --- Launch kernel on a specific GPU ---
    println!("\n--- Kernel Launch on node1:gpu3 ---");
    let device = DeviceId::new(1, 3);
    let kstats = cluster.launch_kernel_on(
        device,
        &kernel,
        &config,
        SchedulingPolicy::Gto,
    );
    println!(
        "  Kernel '{}' on {} | {} threads | occupancy={:.1}%",
        "vec_add",
        device,
        kstats.threads_executed,
        kstats.theoretical_occupancy * 100.0,
    );
}
