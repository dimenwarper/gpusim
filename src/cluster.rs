/// Multi-GPU cluster simulation.
///
/// Models a cluster of nodes, each containing multiple GPUs connected by NVLink,
/// with nodes connected to each other via an InfiniBand fat-tree fabric.
///
/// Topology:
///   Cluster
///   ├── Node 0  (GPUs 0-7, NVLink all-to-all via NVSwitch)
///   ├── Node 1  (GPUs 0-7, NVLink all-to-all via NVSwitch)
///   └── ...
///       connected by InfiniBand fat-tree (NDR/HDR)
use crate::executor::ExecutionStats;
use crate::gpu::GPU;
use crate::interconnect::{
    effective_bandwidth_gb_s, transfer_time_us, AllReduceAlgorithm, CollectiveStats,
    InfiniBandConfig, NVLinkConfig, TransferChannel, TransferStats,
};
use crate::kernel::{Kernel, LaunchConfig};
use crate::metrics::{
    now_ms, read_metrics, write_metrics, CollectiveSnapshot, LiveMetrics, TransferSnapshot,
};
use crate::scheduler::SchedulingPolicy;

// ---------------------------------------------------------------------------
// DeviceId
// ---------------------------------------------------------------------------

/// Identifies a specific GPU in the cluster by (node index, local GPU index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId {
    /// Index of the node within the cluster
    pub node: usize,
    /// Index of the GPU within the node
    pub gpu: usize,
}

impl DeviceId {
    pub fn new(node: usize, gpu: usize) -> Self {
        DeviceId { node, gpu }
    }
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node{}:gpu{}", self.node, self.gpu)
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A single compute node: multiple GPUs connected by NVLink via NVSwitch.
/// All GPUs in a node can communicate at full NVLink bandwidth simultaneously.
pub struct Node {
    pub id: usize,
    /// GPUs on this node
    pub gpus: Vec<GPU>,
    /// NVLink fabric config (all-to-all within node)
    pub nvlink: NVLinkConfig,
}

impl Node {
    pub fn new_h100(id: usize, num_gpus: usize, nvlink: NVLinkConfig) -> Self {
        let gpus = (0..num_gpus).map(|_| GPU::h100()).collect();
        Node { id, gpus, nvlink }
    }
}

// ---------------------------------------------------------------------------
// Cluster
// ---------------------------------------------------------------------------

/// A multi-GPU cluster: multiple nodes connected by an InfiniBand fabric.
pub struct Cluster {
    pub nodes: Vec<Node>,
    /// InfiniBand fabric connecting all nodes
    pub infiniband: InfiniBandConfig,
}

impl Cluster {
    /// Create a cluster with `num_nodes` nodes, each with `gpus_per_node` H100 GPUs.
    pub fn new(
        num_nodes: usize,
        gpus_per_node: usize,
        nvlink: NVLinkConfig,
        infiniband: InfiniBandConfig,
    ) -> Self {
        let nodes = (0..num_nodes)
            .map(|id| Node::new_h100(id, gpus_per_node, nvlink.clone()))
            .collect();
        Cluster { nodes, infiniband }
    }

    /// A standard DGX H100 cluster configuration:
    /// `num_nodes` nodes × 8 H100 GPUs each, NVLink 4.0 + NDR InfiniBand.
    pub fn h100_dgx(num_nodes: usize) -> Self {
        Self::new(num_nodes, 8, NVLinkConfig::h100(), InfiniBandConfig::ndr())
    }

    /// Total number of GPUs in the cluster.
    pub fn total_gpus(&self) -> usize {
        self.nodes.iter().map(|n| n.gpus.len()).sum()
    }

    /// Get an immutable reference to a specific GPU.
    pub fn gpu(&self, device: DeviceId) -> &GPU {
        &self.nodes[device.node].gpus[device.gpu]
    }

    /// Get a mutable reference to a specific GPU.
    pub fn gpu_mut(&mut self, device: DeviceId) -> &mut GPU {
        &mut self.nodes[device.node].gpus[device.gpu]
    }

    // -----------------------------------------------------------------------
    // Point-to-point transfers
    // -----------------------------------------------------------------------

    /// Simulate a point-to-point transfer between two GPUs.
    ///   - Same GPU       → instant (zero time)
    ///   - Same node      → NVLink (high bandwidth, low latency)
    ///   - Different nodes → InfiniBand GPUDirect RDMA (lower bandwidth)
    ///
    /// Writes a cluster metrics snapshot so the live visualizer can show the
    /// transfer activity.
    pub fn transfer(&self, src: DeviceId, dst: DeviceId, bytes: u64) -> TransferStats {
        let result = if src == dst {
            TransferStats::zero(TransferChannel::SameDevice)
        } else {
            let (bandwidth_gb_s, latency_us, channel) = if src.node == dst.node {
                let nv = &self.nodes[src.node].nvlink;
                (nv.bandwidth_gb_s, nv.latency_us, TransferChannel::NVLink)
            } else {
                let ib = &self.infiniband;
                (ib.bandwidth_gb_s, ib.latency_us, TransferChannel::InfiniBand)
            };

            let time_us = transfer_time_us(bytes, bandwidth_gb_s, latency_us);
            let effective_bw = effective_bandwidth_gb_s(bytes, time_us);
            TransferStats { bytes, time_us, effective_bandwidth_gb_s: effective_bw, channel }
        };

        // Write a cluster metrics snapshot for the visualizer
        let mut m = read_metrics().unwrap_or_default();
        self.fill_cluster_header(&mut m);
        m.status = if result.time_us == 0.0 {
            "idle".to_string()
        } else {
            "transfer".to_string()
        };
        m.last_transfer = Some(TransferSnapshot {
            src: src.to_string(),
            dst: dst.to_string(),
            bytes_mb: bytes as f64 / 1_000_000.0,
            time_ms: result.time_us / 1_000.0,
            bandwidth_gb_s: if result.effective_bandwidth_gb_s.is_infinite() {
                0.0
            } else {
                result.effective_bandwidth_gb_s
            },
            channel: result.channel.to_string(),
        });
        m.timestamp_ms = now_ms();
        write_metrics(&m);

        result
    }

    // -----------------------------------------------------------------------
    // Collective operations
    // -----------------------------------------------------------------------

    /// Simulate an AllReduce collective across all GPUs in the cluster.
    ///
    /// The bottleneck link is InfiniBand for multi-node clusters, NVLink for
    /// single-node clusters. The algorithm determines how bandwidth is used.
    pub fn all_reduce(
        &self,
        bytes_per_gpu: u64,
        algorithm: AllReduceAlgorithm,
    ) -> CollectiveStats {
        let n = self.total_gpus();
        let (peak_bw, latency) = self.bottleneck_link();
        let bw_bytes_us = peak_bw * 1_000.0;

        let time_us = match &algorithm {
            AllReduceAlgorithm::Ring => {
                // 2 · (N-1)/N · B/bw  +  2·(N-1)·latency
                2.0 * (n - 1) as f64 / n as f64 * bytes_per_gpu as f64 / bw_bytes_us
                    + 2.0 * (n - 1) as f64 * latency
            }
            AllReduceAlgorithm::Tree => {
                // 2 · ⌈log₂(N)⌉ · (B/bw + latency)
                let steps = (n as f64).log2().ceil() as u32;
                2.0 * steps as f64 * (bytes_per_gpu as f64 / bw_bytes_us + latency)
            }
            AllReduceAlgorithm::Direct => {
                // 2·(N-1) · (B/bw + latency)  — naive reduce-to-root + broadcast
                2.0 * (n - 1) as f64 * (bytes_per_gpu as f64 / bw_bytes_us + latency)
            }
        };

        // NCCL bus bandwidth metric: 2·(N-1)/N · B / time
        let bus_bw = if time_us > 0.0 {
            2.0 * (n - 1) as f64 / n as f64 * bytes_per_gpu as f64 / (time_us * 1_000.0)
        } else {
            0.0
        };

        let stats = CollectiveStats {
            operation: "AllReduce".to_string(),
            algorithm: algorithm.to_string(),
            num_gpus: n,
            bytes_per_gpu,
            time_us,
            bus_bandwidth_gb_s: bus_bw,
            efficiency: (bus_bw / peak_bw).clamp(0.0, 1.0),
        };

        self.write_collective_snapshot(&stats);
        stats
    }

    /// Simulate a Broadcast: one source GPU sends `bytes` to all other GPUs.
    /// Uses a binary-tree algorithm: ⌈log₂(N)⌉ steps.
    pub fn broadcast(&self, src: DeviceId, bytes: u64) -> CollectiveStats {
        let n = self.total_gpus();
        let (peak_bw, latency) = self.bottleneck_link();
        let bw_bytes_us = peak_bw * 1_000.0;
        let steps = (n as f64).log2().ceil() as u32;
        let time_us = steps as f64 * (bytes as f64 / bw_bytes_us + latency);
        let bus_bw = effective_bandwidth_gb_s(bytes, time_us);

        let stats = CollectiveStats {
            operation: format!("Broadcast(src={})", src),
            algorithm: "Tree".to_string(),
            num_gpus: n,
            bytes_per_gpu: bytes,
            time_us,
            bus_bandwidth_gb_s: bus_bw,
            efficiency: (bus_bw / peak_bw).clamp(0.0, 1.0),
        };

        self.write_collective_snapshot(&stats);
        stats
    }

    /// Simulate an AllGather: every GPU ends up with all N chunks concatenated.
    /// Ring algorithm: (N-1) steps, each transferring B bytes per link.
    /// Time ≈ (N-1)/N · B·N / bw  (total data = N·B, pipeline efficiency = (N-1)/N)
    pub fn all_gather(&self, bytes_per_gpu: u64) -> CollectiveStats {
        let n = self.total_gpus();
        let (peak_bw, latency) = self.bottleneck_link();
        let bw_bytes_us = peak_bw * 1_000.0;
        let total_bytes = bytes_per_gpu * n as u64;
        let time_us = (n - 1) as f64 / n as f64 * total_bytes as f64 / bw_bytes_us
            + (n - 1) as f64 * latency;
        let bus_bw = effective_bandwidth_gb_s(total_bytes, time_us);

        let stats = CollectiveStats {
            operation: "AllGather".to_string(),
            algorithm: "Ring".to_string(),
            num_gpus: n,
            bytes_per_gpu,
            time_us,
            bus_bandwidth_gb_s: bus_bw,
            efficiency: (bus_bw / peak_bw).clamp(0.0, 1.0),
        };

        self.write_collective_snapshot(&stats);
        stats
    }

    // -----------------------------------------------------------------------
    // Kernel launch
    // -----------------------------------------------------------------------

    /// Launch a kernel on a specific GPU in the cluster.
    ///
    /// After the kernel completes, the metrics snapshot is enriched with
    /// cluster context (node count, active device, interconnect bandwidth)
    /// so the visualizer can show both kernel stats and cluster topology.
    ///
    /// Transfer and collective history is preserved — the executor writes
    /// `..Default::default()` for cluster fields, so we snapshot them
    /// before launch and restore them afterwards.
    pub fn launch_kernel_on(
        &mut self,
        device: DeviceId,
        kernel: &Kernel,
        config: &LaunchConfig,
        policy: SchedulingPolicy,
    ) -> ExecutionStats {
        // Save any existing transfer/collective snapshots so the executor's
        // ..Default::default() doesn't lose them.
        let prior = read_metrics();
        let saved_transfer = prior.as_ref().and_then(|m| m.last_transfer.clone());
        let saved_collective = prior.as_ref().and_then(|m| m.last_collective.clone());

        let stats =
            self.nodes[device.node].gpus[device.gpu].launch_kernel(kernel, config, policy);

        // Enrich the metrics snapshot written by the executor with cluster
        // context, and restore any transfer/collective history.
        if let Some(mut m) = read_metrics() {
            self.fill_cluster_header(&mut m);
            m.active_device = device.to_string();
            m.last_transfer = m.last_transfer.or(saved_transfer);
            m.last_collective = m.last_collective.or(saved_collective);
            m.timestamp_ms = now_ms();
            write_metrics(&m);
        }

        stats
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Returns the (bandwidth_gb_s, latency_us) of the bottleneck link:
    /// InfiniBand for multi-node clusters, NVLink for single-node.
    fn bottleneck_link(&self) -> (f64, f64) {
        if self.nodes.len() > 1 {
            (self.infiniband.bandwidth_gb_s, self.infiniband.latency_us)
        } else {
            let nv = &self.nodes[0].nvlink;
            (nv.bandwidth_gb_s, nv.latency_us)
        }
    }

    /// Populate cluster-level fields on an existing `LiveMetrics` snapshot.
    fn fill_cluster_header(&self, m: &mut LiveMetrics) {
        m.cluster_mode = true;
        m.num_nodes = self.nodes.len();
        m.gpus_per_node = self.nodes.first().map(|n| n.gpus.len()).unwrap_or(0);
        m.nvlink_bw_gb_s =
            self.nodes.first().map(|n| n.nvlink.bandwidth_gb_s).unwrap_or(0.0);
        m.infiniband_bw_gb_s = self.infiniband.bandwidth_gb_s;
    }

    /// Write a metrics snapshot for a collective operation.
    fn write_collective_snapshot(&self, stats: &CollectiveStats) {
        let mut m = read_metrics().unwrap_or_default();
        self.fill_cluster_header(&mut m);
        m.status = "collective".to_string();
        m.last_collective = Some(CollectiveSnapshot {
            operation: stats.operation.clone(),
            algorithm: stats.algorithm.clone(),
            num_gpus: stats.num_gpus,
            bytes_per_gpu_mb: stats.bytes_per_gpu as f64 / 1_000_000.0,
            time_ms: stats.time_us / 1_000.0,
            bus_bw_gb_s: stats.bus_bandwidth_gb_s,
            efficiency_pct: stats.efficiency * 100.0,
        });
        m.timestamp_ms = now_ms();
        write_metrics(&m);
    }
}
