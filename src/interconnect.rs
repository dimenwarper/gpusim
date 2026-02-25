/// Communication channel models for multi-GPU clusters.
///
/// Models NVLink (intra-node, via NVSwitch) and InfiniBand (inter-node, fat-tree)
/// interconnects, including point-to-point transfer time simulation and
/// collective operation algorithms (Ring, Tree, Direct AllReduce).
///
/// Bandwidth reference:
///   H100 NVLink 4.0 — 900 GB/s bidirectional per GPU (via NVSwitch)
///   A100 NVLink 3.0 — 600 GB/s bidirectional per GPU
///   NDR InfiniBand  — 400 Gb/s = 50 GB/s per link
///   HDR InfiniBand  — 200 Gb/s = 25 GB/s per link

// ---------------------------------------------------------------------------
// Channel configurations
// ---------------------------------------------------------------------------

/// NVLink configuration — intra-node all-to-all via NVSwitch.
/// All GPUs in a node can communicate at full bandwidth simultaneously.
#[derive(Debug, Clone)]
pub struct NVLinkConfig {
    /// Peak bidirectional bandwidth per GPU in GB/s
    pub bandwidth_gb_s: f64,
    /// Point-to-point latency in microseconds
    pub latency_us: f64,
}

impl NVLinkConfig {
    /// H100 SXM: NVLink 4.0 — 900 GB/s bidirectional per GPU via NVSwitch
    pub fn h100() -> Self {
        NVLinkConfig { bandwidth_gb_s: 900.0, latency_us: 1.0 }
    }

    /// A100 SXM: NVLink 3.0 — 600 GB/s bidirectional per GPU
    pub fn a100() -> Self {
        NVLinkConfig { bandwidth_gb_s: 600.0, latency_us: 1.0 }
    }
}

/// InfiniBand fabric configuration — inter-node, fat-tree topology.
/// Fat-tree provides non-blocking full bisection bandwidth for uniform traffic.
#[derive(Debug, Clone)]
pub struct InfiniBandConfig {
    /// Per-link bandwidth in GB/s
    pub bandwidth_gb_s: f64,
    /// End-to-end latency in microseconds (including switch hops)
    pub latency_us: f64,
    /// Number of fat-tree spine levels (typically 2 for HPC clusters)
    pub spine_levels: u32,
}

impl InfiniBandConfig {
    /// HDR InfiniBand — 200 Gb/s = 25 GB/s per link
    pub fn hdr() -> Self {
        InfiniBandConfig { bandwidth_gb_s: 25.0, latency_us: 2.0, spine_levels: 2 }
    }

    /// NDR InfiniBand — 400 Gb/s = 50 GB/s per link
    pub fn ndr() -> Self {
        InfiniBandConfig { bandwidth_gb_s: 50.0, latency_us: 2.0, spine_levels: 2 }
    }

    /// XDR InfiniBand — 800 Gb/s = 100 GB/s per link (next generation)
    pub fn xdr() -> Self {
        InfiniBandConfig { bandwidth_gb_s: 100.0, latency_us: 1.5, spine_levels: 2 }
    }
}

// ---------------------------------------------------------------------------
// Transfer simulation
// ---------------------------------------------------------------------------

/// Which physical channel was used for a transfer.
#[derive(Debug, Clone, PartialEq)]
pub enum TransferChannel {
    /// No transfer needed (src == dst)
    SameDevice,
    /// Intra-node via NVLink / NVSwitch
    NVLink,
    /// Inter-node via InfiniBand (GPUDirect RDMA)
    InfiniBand,
}

impl std::fmt::Display for TransferChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransferChannel::SameDevice   => write!(f, "same-device"),
            TransferChannel::NVLink       => write!(f, "NVLink"),
            TransferChannel::InfiniBand   => write!(f, "InfiniBand"),
        }
    }
}

/// Result of a simulated point-to-point transfer.
#[derive(Debug, Clone)]
pub struct TransferStats {
    /// Number of bytes transferred
    pub bytes: u64,
    /// Simulated transfer time in microseconds
    pub time_us: f64,
    /// Effective bandwidth achieved in GB/s
    pub effective_bandwidth_gb_s: f64,
    /// Channel used
    pub channel: TransferChannel,
}

impl TransferStats {
    pub fn zero(channel: TransferChannel) -> Self {
        TransferStats {
            bytes: 0,
            time_us: 0.0,
            effective_bandwidth_gb_s: f64::INFINITY,
            channel,
        }
    }
}

/// Compute simulated transfer time in microseconds.
///
/// time_µs = latency_µs + bytes / bandwidth_bytes_per_µs
///
/// Unit note: 1 GB/s = 10⁹ bytes/s = 10³ bytes/µs
pub fn transfer_time_us(bytes: u64, bandwidth_gb_s: f64, latency_us: f64) -> f64 {
    if bytes == 0 { return 0.0; }
    let bandwidth_bytes_per_us = bandwidth_gb_s * 1_000.0;
    latency_us + bytes as f64 / bandwidth_bytes_per_us
}

/// Compute effective bandwidth in GB/s from bytes and transfer time.
pub fn effective_bandwidth_gb_s(bytes: u64, time_us: f64) -> f64 {
    if time_us == 0.0 { return f64::INFINITY; }
    bytes as f64 / time_us / 1_000.0
}

// ---------------------------------------------------------------------------
// Collective operation algorithms
// ---------------------------------------------------------------------------

/// AllReduce algorithm selection.
///
/// AllReduce semantics: every GPU starts with a local tensor of `bytes_per_gpu`
/// bytes; after the operation every GPU holds the element-wise sum across all GPUs.
#[derive(Debug, Clone)]
pub enum AllReduceAlgorithm {
    /// Ring-AllReduce (Baidu/NCCL default for large messages).
    /// Two phases (reduce-scatter + all-gather), each (N-1) steps.
    /// Bandwidth-optimal: approaches 100% link utilization for large N.
    /// Time ≈ 2·(N-1)/N · B/bw  +  2·(N-1)·latency
    Ring,

    /// Recursive halving/doubling (tree reduce + broadcast).
    /// Latency-optimal: only log₂(N) steps.
    /// Better than Ring for small messages; worse for large.
    /// Time ≈ 2·⌈log₂(N)⌉ · (B/bw + latency)
    Tree,

    /// Naive direct: all GPUs reduce to rank 0, then rank 0 broadcasts.
    /// Poor scalability but simple baseline.
    /// Time ≈ 2·(N-1)·(B/bw + latency)
    Direct,
}

impl std::fmt::Display for AllReduceAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllReduceAlgorithm::Ring   => write!(f, "Ring"),
            AllReduceAlgorithm::Tree   => write!(f, "Tree"),
            AllReduceAlgorithm::Direct => write!(f, "Direct"),
        }
    }
}

/// Result of a simulated collective operation.
#[derive(Debug, Clone)]
pub struct CollectiveStats {
    /// Name of the operation (e.g. "AllReduce", "Broadcast")
    pub operation: String,
    /// Algorithm used
    pub algorithm: String,
    /// Number of GPUs participating
    pub num_gpus: usize,
    /// Bytes per GPU (payload each GPU contributes)
    pub bytes_per_gpu: u64,
    /// Simulated wall-clock time in microseconds
    pub time_us: f64,
    /// Effective bus bandwidth in GB/s
    /// (AllReduce: 2·(N-1)/N·B / time — the NCCL bus bandwidth metric)
    pub bus_bandwidth_gb_s: f64,
    /// Bandwidth efficiency: bus_bw / peak_link_bw [0.0, 1.0]
    pub efficiency: f64,
}
