/// Live metrics for the TUI visualizer.
///
/// The executor writes a JSON snapshot to METRICS_PATH after every block
/// execution. The viz binary polls this file and re-renders the dashboard.
/// Writes are atomic (write to .tmp then rename) to avoid torn reads.
///
/// Cluster operations (transfer, collective) also write snapshots so the
/// visualizer can show interconnect activity alongside kernel execution.
use serde::{Deserialize, Serialize};

pub const METRICS_PATH: &str = "/tmp/gpusim_live.json";

// ---------------------------------------------------------------------------
// Cluster snapshot types
// ---------------------------------------------------------------------------

/// Snapshot of a point-to-point transfer between two GPUs.
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct TransferSnapshot {
    /// Source device, e.g. "node0:gpu0"
    pub src: String,
    /// Destination device, e.g. "node0:gpu1"
    pub dst: String,
    /// Data volume in MB
    pub bytes_mb: f64,
    /// Transfer time in milliseconds
    pub time_ms: f64,
    /// Effective bandwidth in GB/s
    pub bandwidth_gb_s: f64,
    /// Channel name: "NVLink", "InfiniBand", or "same-device"
    pub channel: String,
}

/// Snapshot of a collective operation (AllReduce, AllGather, Broadcast, …).
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct CollectiveSnapshot {
    /// Operation name, e.g. "AllReduce"
    pub operation: String,
    /// Algorithm name, e.g. "Ring"
    pub algorithm: String,
    /// Number of GPUs participating
    pub num_gpus: usize,
    /// Payload per GPU in MB
    pub bytes_per_gpu_mb: f64,
    /// Wall time in milliseconds
    pub time_ms: f64,
    /// Bus bandwidth in GB/s (NCCL-style metric)
    pub bus_bw_gb_s: f64,
    /// Efficiency as a percentage [0, 100]
    pub efficiency_pct: f64,
}

// ---------------------------------------------------------------------------
// LiveMetrics
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct LiveMetrics {
    // ------------------------------------------------------------------
    // Single-GPU kernel fields
    // ------------------------------------------------------------------

    /// "idle" | "running" | "complete" | "transfer" | "collective"
    pub status: String,
    pub kernel_name: String,
    pub scheduling_policy: String,
    /// Grid dimensions [x, y, z]
    pub grid: [u32; 3],
    /// Block dimensions [x, y, z]
    pub block: [u32; 3],
    /// Theoretical occupancy [0.0, 1.0]
    pub theoretical_occupancy: f32,
    /// Which resource limited occupancy
    pub occupancy_limiter: String,
    /// Max blocks that can fit per SM
    pub max_blocks_per_sm: u32,
    /// Total blocks in the grid
    pub blocks_total: u32,
    /// Blocks completed so far
    pub blocks_executed: u32,
    /// Warps executed so far
    pub warps_executed: u32,
    /// Threads executed so far
    pub threads_executed: u32,
    /// Active block count per SM — index = SM id
    pub sm_active_blocks: Vec<u32>,
    /// Unix timestamp in ms when this snapshot was written
    pub timestamp_ms: u64,

    // ------------------------------------------------------------------
    // Cluster fields — only populated in multi-GPU cluster mode.
    // All use #[serde(default)] so old single-GPU snapshots deserialize fine.
    // ------------------------------------------------------------------

    /// Whether this is a cluster (multi-GPU) simulation
    #[serde(default)]
    pub cluster_mode: bool,
    /// Number of nodes in the cluster
    #[serde(default)]
    pub num_nodes: usize,
    /// Number of GPUs per node
    #[serde(default)]
    pub gpus_per_node: usize,
    /// NVLink peak bandwidth in GB/s (intra-node)
    #[serde(default)]
    pub nvlink_bw_gb_s: f64,
    /// InfiniBand peak bandwidth in GB/s (inter-node)
    #[serde(default)]
    pub infiniband_bw_gb_s: f64,
    /// Which GPU is (or last was) running the kernel, e.g. "node1:gpu3".
    /// Empty string when not in cluster mode.
    #[serde(default)]
    pub active_device: String,
    /// Most recent point-to-point transfer (if any)
    #[serde(default)]
    pub last_transfer: Option<TransferSnapshot>,
    /// Most recent collective operation (if any)
    #[serde(default)]
    pub last_collective: Option<CollectiveSnapshot>,
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

/// Atomically write metrics to METRICS_PATH.
/// Uses a .tmp intermediate file + rename to avoid torn reads by the viz.
pub fn write_metrics(metrics: &LiveMetrics) {
    if let Ok(json) = serde_json::to_string(metrics) {
        let tmp = format!("{}.tmp", METRICS_PATH);
        if std::fs::write(&tmp, &json).is_ok() {
            let _ = std::fs::rename(&tmp, METRICS_PATH);
        }
    }
}

/// Read the latest metrics snapshot. Returns None if the file doesn't exist
/// or can't be parsed (e.g. no simulation has run yet).
pub fn read_metrics() -> Option<LiveMetrics> {
    let data = std::fs::read_to_string(METRICS_PATH).ok()?;
    serde_json::from_str(&data).ok()
}

/// Returns current Unix time in milliseconds.
pub fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
