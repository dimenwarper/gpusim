# gpusim

A GPU simulator written in Rust, modelling the architecture of modern NVIDIA GPUs (H100/Hopper) from the SM/warp/thread level up to multi-GPU clusters connected by NVLink and InfiniBand. The ultimate goal is to execute kernels and study scheduling, occupancy, memory hierarchy, and distributed communication behaviour.

> Architecture based on: [JAX Scaling Book — GPUs](https://jax-ml.github.io/scaling-book/gpus/), NVIDIA architecture whitepapers, and [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution).

---

## Architecture Overview

Modern GPU clusters span three levels of hierarchy — from individual compute units inside a GPU, up through multi-GPU nodes, to multi-node clusters. `gpusim` models all three:

```
Cluster
├── Node 0  ─────────────────────── NVLink 4.0 (900 GB/s, all-to-all)
│   ├── GPU 0
│   │   ├── SMs (×132)
│   │   │   ├── SMEM           — 256KB fast on-chip shared memory per SM
│   │   │   ├── Warp Schedulers (×4 subpartitions)
│   │   │   └── Tensor Cores   — matrix multiply-accumulate (MMA) units
│   │   ├── L2 Cache           — ~50MB shared across all SMs
│   │   └── HBM                — 80GB high-bandwidth memory (3.4 TB/s)
│   └── GPU 1 … GPU 7
├── Node 1  ─────────────────────── NVLink 4.0
│   └── GPU 0 … GPU 7
└── InfiniBand fabric (NDR 400Gb/s, fat-tree) connecting all nodes
```

Kernels are launched with a **grid** of **thread blocks**. Each block is assigned to an SM and its threads are divided into **warps** of 32. The warp scheduler determines execution order within the SM. Across GPUs, communication is modelled via NVLink (intra-node) and InfiniBand (inter-node) with realistic bandwidth and latency parameters.

---

## Features

### Kernel Execution
- CUDA-style execution model: `grid_dim`, `block_dim`, `threadIdx`, `blockIdx`
- Per-thread `ThreadCtx` with access to SMEM and global HBM memory
- Kernels defined as Rust closures — no DSL or bytecode needed

### Block Scheduling (GigaThread Engine)
- Blocks assigned to the SM with the **most available resource headroom**, matching empirically observed NVIDIA behaviour (Gilman et al., SIGMETRICS 2021)
- Resource-aware: tracks threads, warps, registers, and SMEM per SM

### Occupancy Calculation
- Full 5-limiter occupancy model (mirrors GPGPU-Sim's `max_cta()` logic):
  1. Thread slots
  2. Warp slots
  3. Register file (with allocation granularity)
  4. Shared memory (with allocation granularity)
  5. Hardware block cap
- Reports theoretical occupancy and identifies the bottleneck resource

### Warp Scheduling (3 policies)

| Policy | Description | Best for |
|---|---|---|
| **LRR** | Loose Round-Robin — rotates through warps equally | Baseline comparison |
| **GTO** | Greedy-Then-Oldest — sticks with one warp until it stalls, then picks the oldest eligible | Cache locality (default in GPGPU-Sim) |
| **TwoLevel** | Active set + pending pool — LRR within active set, promotes from pending when stalled | Best overall (~19% over LRR, Narasiman et al. MICRO 2011) |

### Memory Hierarchy
- **SMEM** — per-block on-chip scratch memory (256KB per SM)
- **L2 Cache** — shared across all SMs (~50MB), sparse-mapped
- **HBM** — 80GB main memory, sparse-mapped (no eager 80GB allocation on host)

### SM Configurations
- `SmConfig::h100()` — Hopper (CC 9.0): 132 SMs, 64 warps/SM, 228KB SMEM/SM
- `SmConfig::a100()` — Ampere (CC 8.0): 164KB SMEM/SM

### Multi-GPU Clusters
- `Cluster` — N nodes × M GPUs, NVLink intra-node + InfiniBand inter-node
- **Point-to-point transfers** routed automatically: NVLink if same node, InfiniBand if cross-node
- **Collective operations**: AllReduce (Ring / Tree / Direct), AllGather (Ring), Broadcast (Tree)
- Bandwidth and latency modelled realistically; reports time, effective bandwidth, and efficiency
- `launch_kernel_on(device, kernel, config, policy)` — run a kernel on any GPU in the cluster

| Interconnect | Preset | Bandwidth | Latency |
|---|---|---|---|
| NVLink 4.0 (H100) | `NVLinkConfig::h100()` | 900 GB/s | 1 µs |
| NVLink 3.0 (A100) | `NVLinkConfig::a100()` | 600 GB/s | 1 µs |
| NDR InfiniBand | `InfiniBandConfig::ndr()` | 50 GB/s | 2 µs |
| HDR InfiniBand | `InfiniBandConfig::hdr()` | 25 GB/s | 2 µs |
| XDR InfiniBand | `InfiniBandConfig::xdr()` | 100 GB/s | 1.5 µs |

### Live TUI Visualizer
- Attach to any running simulation from a **separate terminal at any time** — no need to restart the sim
- Polls `/tmp/gpusim_live.json` every 200ms (written atomically by the executor after each block)
- Shows SM utilization heatmap, occupancy gauge, block progress, and kernel stats
- Press `q` or `Esc` to quit; the simulation keeps running unaffected

---

## Module Structure

```
src/
├── main.rs         — Entry point; single-GPU and multi-GPU demos
├── lib.rs          — Module declarations
├── gpu.rs          — Top-level GPU struct; launch_kernel()
├── sm.rs           — StreamingMultiprocessor; resource tracking
├── kernel.rs       — Dim3, LaunchConfig, ThreadCtx, Kernel
├── executor.rs     — KernelExecutor; block + warp scheduling loop; metrics snapshots
├── occupancy.rs    — SmConfig, KernelResources, max_blocks_per_sm()
├── scheduler.rs    — WarpState, WarpSlot, LRR/GTO/TwoLevel schedulers
├── metrics.rs      — LiveMetrics; atomic write/read to /tmp/gpusim_live.json
├── memory.rs       — L2Cache and HBM (sparse HashMap-backed)
├── warp.rs         — Warp struct (registers, PC, age)
├── tensor_core.rs  — TensorCore MMA unit
├── cluster.rs      — Cluster, Node, DeviceId; transfer(), all_reduce(), all_gather()
├── interconnect.rs — NVLinkConfig, InfiniBandConfig, transfer math, collective algorithms
└── bin/
    └── viz.rs      — Live TUI visualizer (ratatui)
```

---

## Usage

### Running the demo

```bash
cargo run
```

Output:
```
Initialized H100-like GPU with 132 SMs, 50MB L2 cache, 80GB HBM
[gpusim] Launching kernel 'vec_add' | grid=(8,1,1) block=(128,1,1) | policy=GTO | max_blocks/SM=16 | occupancy=100.0% (limited by register file)
[gpusim] Kernel 'vec_add' complete | 8 blocks | 32 warps | 1024 threads | occupancy=100.0%
Verification PASSED: all 1024 results correct (each = 1024)
Stats: 8 blocks | 32 warps | 1024 threads | occupancy=100.0% (limited by register file) | policy=GTO
```

### Writing a kernel

```rust
use gpusim::gpu::GPU;
use gpusim::kernel::{Dim3, Kernel, LaunchConfig};
use gpusim::scheduler::SchedulingPolicy;

let mut gpu = GPU::h100();

// Define a kernel — doubles every element in-place
let kernel = Kernel::new("double", |ctx| {
    let addr = ctx.global_id() as usize * 4;
    let val = f32::from_le_bytes(ctx.gmem.read(addr, 4).try_into().unwrap());
    ctx.gmem.write(addr, &(val * 2.0).to_le_bytes());
});

// Launch: 128 threads/block, 32 regs/thread for occupancy calculation
let config = LaunchConfig::new(Dim3::x(8), Dim3::x(128))
    .with_resources(32, 0);

let stats = gpu.launch_kernel(&kernel, &config, SchedulingPolicy::Gto);

println!("Occupancy: {:.1}%", stats.theoretical_occupancy * 100.0);
println!("Bottleneck: {}", stats.occupancy_limiter);
```

### Choosing a scheduling policy

```rust
// Loose Round-Robin
gpu.launch_kernel(&kernel, &config, SchedulingPolicy::Lrr);

// Greedy-Then-Oldest (default in GPGPU-Sim)
gpu.launch_kernel(&kernel, &config, SchedulingPolicy::Gto);

// Two-Level with active set of 8 warps
gpu.launch_kernel(&kernel, &config, SchedulingPolicy::TwoLevel { active_set_size: 8 });
```

### Multi-GPU cluster

```rust
use gpusim::cluster::{Cluster, DeviceId};
use gpusim::interconnect::AllReduceAlgorithm;

// 2 nodes × 8 H100s = 16 GPUs, NVLink 4.0 + NDR InfiniBand
let mut cluster = Cluster::h100_dgx(2);

// Point-to-point transfer: 1GB intra-node (NVLink)
let t = cluster.transfer(DeviceId::new(0, 0), DeviceId::new(0, 1), 1 << 30);
println!("{:.2}ms  ({:.0} GB/s)", t.time_us / 1000.0, t.effective_bandwidth_gb_s);
// → 1.19ms  (899 GB/s)

// Point-to-point transfer: 1GB inter-node (InfiniBand)
let t = cluster.transfer(DeviceId::new(0, 0), DeviceId::new(1, 0), 1 << 30);
println!("{:.2}ms  ({:.0} GB/s)", t.time_us / 1000.0, t.effective_bandwidth_gb_s);
// → 21.48ms  (50 GB/s)

// AllReduce: 1GB per GPU across all 16 GPUs
let s = cluster.all_reduce(1 << 30, AllReduceAlgorithm::Ring);
println!("{:.2}ms  efficiency={:.1}%", s.time_us / 1000.0, s.efficiency * 100.0);
// → 40.33ms  efficiency=99.9%

// Launch a kernel on a specific GPU
cluster.launch_kernel_on(DeviceId::new(1, 3), &kernel, &config, SchedulingPolicy::Gto);
```

### Live visualizer

Start the simulation in one terminal, then attach the visualizer in another at any time:

```bash
# Terminal 1 — run simulation
cargo run

# Terminal 2 — attach visualizer (start before, during, or after)
cargo run --bin viz
```

The visualizer shows:

```
┌─ ⚡ gpusim live monitor ─────────────────────────────────────────────────────┐
│  kernel: vec_add   policy: GTO   status: RUNNING                             │
├──────────────────────────────────────┬───────────────────────────────────────┤
│ SM Utilization                       │ Stats                                 │
│ ██ active   ░░ idle                  │ Occupancy  ████████████████ 100.0%    │
│                                      │ Blocks     ████░░░░░░░░░░░░  3 / 8   │
│ ██ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░   │                                       │
│ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░   │ Warps:      12                        │
│ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░   │ Threads:    384                       │
│ ...                                  │ Max blk/SM: 16                        │
│  1/132 SMs active                    │ Limiter:    register file             │
│                                      │ Grid:   (8,1,1)                       │
│                                      │ Block:  (128,1,1)                     │
├──────────────────────────────────────┴───────────────────────────────────────┤
│  q / esc: quit    auto-refreshes every 200ms    reads /tmp/gpusim_live.json  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Key References

| Topic | Reference |
|---|---|
| GPU architecture | [JAX Scaling Book — GPUs](https://jax-ml.github.io/scaling-book/gpus/) |
| Block scheduling | [Gilman et al., SIGMETRICS 2021 — Demystifying NVIDIA's TBS](https://dl.acm.org/doi/10.1145/3453953.3453472) |
| Fermi TBS details | [Sreepathi Pai — How the Fermi TBS Works](https://www.cs.rochester.edu/~sree/fermi-tbs/fermi-tbs.html) |
| GTO scheduling | [Rogers et al., MICRO 2012 — Cache-Conscious Wavefront Scheduling](https://engineering.purdue.edu/tgrogers/publication/rogers-micro-2012/rogers-micro-2012.pdf) |
| Two-level scheduling | [Narasiman et al., MICRO 2011 — Two-Level Warp Scheduling](https://users.ece.cmu.edu/~omutlu/pub/large-gpu-warps_micro11.pdf) |
| Latency hiding | [Volkov, 2016 — Understanding Latency Hiding on GPUs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf) |
| Occupancy formula | [Lei Mao — CUDA Occupancy Calculation](https://leimao.github.io/blog/CUDA-Occupancy-Calculation/) |
| Reference simulator | [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) |

---

## Roadmap

- [ ] Warp stall simulation (LongScoreboard / ShortScoreboard state transitions)
- [ ] Cycle-accurate execution with simulated memory latency
- [ ] Python bindings for running experiments from Python notebooks
- [ ] Matrix multiplication kernel demo (using TensorCore MMA)
- [ ] Multi-kernel / concurrent kernel execution
- [ ] Performance metrics: IPC, memory bandwidth utilisation, stall breakdown
- [ ] Richer visualizer: warp state timeline, per-SM drill-down, stall breakdown chart
- [ ] Multi-GPU visualizer: per-node/per-GPU view, transfer activity, collective progress

---

## License

MIT
