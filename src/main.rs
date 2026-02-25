use gpusim::gpu::GPU;

fn main() {
    let gpu = GPU::h100();
    println!(
        "Initialized H100-like GPU with {} SMs, {}MB L2 cache, {}GB HBM",
        gpu.sms.len(),
        gpu.l2_cache.size_bytes / (1024 * 1024),
        gpu.hbm.size_bytes / (1024 * 1024 * 1024),
    );
}
