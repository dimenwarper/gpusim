# gpusim

A GPU simulator written in Rust, modelling the architecture of modern NVIDIA GPUs (H100/Hopper) down to the SM, warp, and thread level. The ultimate goal is to execute kernels and study scheduling, occupancy, and memory hierarchy behaviour.

> Architecture based on: [JAX Scaling Book — GPUs](https://jax-ml.github.io/scaling-book/gpus/), NVIDIA architecture whitepapers, and [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution).

---

## Architecture Overview

Modern GPUs are made up of hundreds of **Streaming Multiprocessors (SMs)** connected to a shared memory hierarchy. `gpusim` models this at three levels:

```
GPU
├── SMs (×132 on H100)
│   ├── SMEM           — 256KB fast on-chip shared memory per SM
│   ├── Warp Schedulers (×4 subpartitions)
│   └── Tensor Cores   — matrix multiply-accumulate (MMA) units
├── L2 Cache           — ~50MB shared across all SMs
└── HBM                — 80GB high-bandwidth main memory (3.4 TB/s)
```

Kernels are launched with a **grid** of **thread blocks**. Each block is assigned to an SM and its threads are divided into **warps** of 32. The warp scheduler determines execution order within the SM.

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

---

## Module Structure

```
src/
├── main.rs         — Entry point; vec_add demo kernel
├── lib.rs          — Module declarations
├── gpu.rs          — Top-level GPU struct; launch_kernel()
├── sm.rs           — StreamingMultiprocessor; resource tracking
├── kernel.rs       — Dim3, LaunchConfig, ThreadCtx, Kernel
├── executor.rs     — KernelExecutor; block + warp scheduling loop
├── occupancy.rs    — SmConfig, KernelResources, max_blocks_per_sm()
├── scheduler.rs    — WarpState, WarpSlot, LRR/GTO/TwoLevel schedulers
├── memory.rs       — L2Cache and HBM (sparse HashMap-backed)
├── warp.rs         — Warp struct (registers, PC, age)
└── tensor_core.rs  — TensorCore MMA unit
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

---

## License

MIT
