/// Live metrics for the TUI visualizer.
///
/// The executor writes a JSON snapshot to METRICS_PATH after every block
/// execution. The viz binary polls this file and re-renders the dashboard.
/// Writes are atomic (write to .tmp then rename) to avoid torn reads.
use serde::{Deserialize, Serialize};

pub const METRICS_PATH: &str = "/tmp/gpusim_live.json";

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct LiveMetrics {
    /// "idle" | "running" | "complete"
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
}

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
