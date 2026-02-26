use gpusim::cluster::{Cluster, DeviceId};
use gpusim::gpu::GPU;
use gpusim::interconnect::AllReduceAlgorithm;
use gpusim::kernel::{Dim3, Kernel, LaunchConfig};
use gpusim::scheduler::SchedulingPolicy;
use std::thread::sleep;
use std::time::Duration;

fn main() {
    // -----------------------------------------------------------------------
    // Single-GPU vector addition demo
    // -----------------------------------------------------------------------
    let mut gpu = GPU::h100();
    println!(
        "Initialized H100-like GPU with {} SMs, {}MB L2 cache, {}GB HBM",
        gpu.sms.len(),
        gpu.l2_cache.size_bytes / (1024 * 1024),
        gpu.hbm.size_bytes / (1024 * 1024 * 1024),
    );

    // 32 768 elements → 256 blocks of 128 threads.
    // Large enough that the SM heatmap has time to animate in the visualiser.
    let n: u32 = 32_768;
    let stride = std::mem::size_of::<f32>();

    let base_a: usize = 0;
    let base_b: usize = n as usize * stride;
    let base_c: usize = 2 * n as usize * stride;

    for i in 0..n as usize {
        let a_val: f32 = i as f32;
        let b_val: f32 = (n as usize - i) as f32;
        gpu.hbm.write(base_a + i * stride, &a_val.to_le_bytes());
        gpu.hbm.write(base_b + i * stride, &b_val.to_le_bytes());
    }

    let kernel = Kernel::new("vec_add", move |ctx| {
        let i = ctx.global_id() as usize;
        if i >= n as usize {
            return;
        }
        let a = f32::from_le_bytes(ctx.gmem.read(base_a + i * stride, 4).try_into().unwrap());
        let b = f32::from_le_bytes(ctx.gmem.read(base_b + i * stride, 4).try_into().unwrap());
        ctx.gmem.write(base_c + i * stride, &(a + b).to_le_bytes());
    });

    let threads_per_block = 128u32;
    let num_blocks = n.div_ceil(threads_per_block);

    // 60 ms per block  ×  256 blocks  ≈  15 s — plenty of time to watch the heatmap.
    // Remove .with_delay() to run at full speed without visualization.
    let config = LaunchConfig::new(Dim3::x(num_blocks), Dim3::x(threads_per_block))
        .with_resources(32, 0)
        .with_delay(60);

    let stats = gpu.launch_kernel(&kernel, &config, SchedulingPolicy::Gto);

    // Verify results (single-GPU run only)
    let mut all_correct = true;
    for i in 0..n as usize {
        let bytes = gpu.hbm.read(base_c + i * stride, 4);
        let c = f32::from_le_bytes(bytes.try_into().unwrap());
        if (c - n as f32).abs() > 1e-5 {
            println!("MISMATCH at i={}: got {}, expected {}", i, c, n);
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
    // Multi-GPU cluster demo — runs in an infinite loop so the visualiser
    // always has fresh data to display.  Press Ctrl-C to stop.
    // -----------------------------------------------------------------------
    println!("\n{}", "=".repeat(60));
    println!("Multi-GPU Cluster Demo  (loops indefinitely — Ctrl-C to stop)");
    println!("{}", "=".repeat(60));

    let one_gb = 1u64 << 30;
    let device = DeviceId::new(1, 3);

    // Config used for cluster kernel launches — same delay as single-GPU demo.
    let cluster_config = LaunchConfig::new(Dim3::x(num_blocks), Dim3::x(threads_per_block))
        .with_resources(32, 0)
        .with_delay(60);

    loop {
        let mut cluster = Cluster::h100_dgx(2);
        println!(
            "\nCluster: {} nodes × {} GPUs = {} GPUs  \
             NVLink {:.0} GB/s │ IB {:.0} GB/s",
            cluster.nodes.len(),
            cluster.nodes[0].gpus.len(),
            cluster.total_gpus(),
            cluster.nodes[0].nvlink.bandwidth_gb_s,
            cluster.infiniband.bandwidth_gb_s,
        );

        // --- Point-to-point transfers ---
        println!("  [transfer] same-device …");
        cluster.transfer(DeviceId::new(0, 0), DeviceId::new(0, 0), one_gb);
        sleep(Duration::from_millis(800));

        println!("  [transfer] NVLink (intra-node) …");
        cluster.transfer(DeviceId::new(0, 0), DeviceId::new(0, 1), one_gb);
        sleep(Duration::from_millis(800));

        println!("  [transfer] InfiniBand (inter-node) …");
        cluster.transfer(DeviceId::new(0, 0), DeviceId::new(1, 0), one_gb);
        sleep(Duration::from_millis(800));

        // --- AllReduce variants ---
        for algo in [AllReduceAlgorithm::Ring, AllReduceAlgorithm::Tree, AllReduceAlgorithm::Direct] {
            println!("  [collective] AllReduce/{} …", algo);
            let s = cluster.all_reduce(one_gb, algo);
            println!(
                "    → {:.2}ms  {:.1} GB/s bus BW  {:.1}% efficiency",
                s.time_us / 1_000.0,
                s.bus_bandwidth_gb_s,
                s.efficiency * 100.0,
            );
            sleep(Duration::from_millis(1_000));
        }

        // --- AllGather ---
        println!("  [collective] AllGather …");
        let s = cluster.all_gather(128 * 1024 * 1024);
        println!(
            "    → {:.2}ms  {:.1} GB/s bus BW",
            s.time_us / 1_000.0,
            s.bus_bandwidth_gb_s,
        );
        sleep(Duration::from_millis(1_000));

        // --- Kernel launch on a specific GPU ---
        println!("  [kernel] vec_add on {} …", device);
        let ks = cluster.launch_kernel_on(device, &kernel, &cluster_config, SchedulingPolicy::Gto);
        println!(
            "    → {} threads | occupancy={:.1}%",
            ks.threads_executed,
            ks.theoretical_occupancy * 100.0,
        );

        // Brief pause before the next round
        println!("  — round complete, restarting in 2 s —");
        sleep(Duration::from_secs(2));
    }
}
