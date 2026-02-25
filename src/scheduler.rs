/// Warp scheduling policies.
///
/// Each cycle, the warp scheduler selects which eligible warp to issue next.
/// Three policies are implemented, matching those studied in the literature:
///   - LRR  (Loose Round-Robin)       — simple rotation, baseline
///   - GTO  (Greedy-Then-Oldest)      — cache-friendly, default in GPGPU-Sim
///   - TwoLevel (Two-Level Active)    — active set + pending pool, best overall
///
/// References:
///   Narasiman et al., MICRO 2011 — Two-Level Warp Scheduling
///   Rogers, O'Connor, Aamodt, MICRO 2012 — GTO / Cache-Conscious Scheduling

/// The execution state of a warp. Mirrors NVIDIA Nsight Compute stall taxonomy.
#[derive(Debug, Clone, PartialEq)]
pub enum WarpState {
    /// Ready to issue: no unsatisfied dependencies, functional unit available.
    Eligible,
    /// Waiting on a global/HBM memory load (~400-800 simulated cycles).
    LongScoreboard,
    /// Waiting on shared memory or fixed-latency arithmetic (~10-30 cycles).
    ShortScoreboard,
    /// Waiting at a __syncthreads() barrier for other warps in the block.
    Barrier,
    /// Waiting for result of a prior instruction from the same warp (RAW hazard).
    ExecDep,
    /// Memory subsystem request queue is full; cannot issue new loads/stores.
    MemThrottle,
    /// Waiting for instruction cache to return the next instruction.
    Fetch,
    /// No more instructions; warp has exited.
    Idle,
}

impl WarpState {
    pub fn is_eligible(&self) -> bool {
        *self == WarpState::Eligible
    }
}

impl std::fmt::Display for WarpState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WarpState::Eligible        => write!(f, "Eligible"),
            WarpState::LongScoreboard  => write!(f, "LongScoreboard"),
            WarpState::ShortScoreboard => write!(f, "ShortScoreboard"),
            WarpState::Barrier         => write!(f, "Barrier"),
            WarpState::ExecDep         => write!(f, "ExecDep"),
            WarpState::MemThrottle     => write!(f, "MemThrottle"),
            WarpState::Fetch           => write!(f, "Fetch"),
            WarpState::Idle            => write!(f, "Idle"),
        }
    }
}

/// The scheduler's view of a warp — its index, current state, and age.
/// The executor creates these from the active warps in a block.
#[derive(Debug, Clone)]
pub struct WarpSlot {
    /// Warp index within the block
    pub warp_idx: usize,
    /// Current execution state
    pub state: WarpState,
    /// Launch timestamp — lower = older. Used for GTO age-based fallback.
    pub age: u64,
}

impl WarpSlot {
    pub fn new(warp_idx: usize, age: u64) -> Self {
        WarpSlot {
            warp_idx,
            state: WarpState::Eligible,
            age,
        }
    }
}

/// Trait for warp scheduling policies.
pub trait WarpScheduler: Send {
    /// Given the current warp slots, return a priority-ordered list of warp indices.
    /// The executor issues the first eligible warp in this list each cycle.
    fn order_warps(&mut self, slots: &[WarpSlot]) -> Vec<usize>;

    /// Called after a warp is successfully issued so the scheduler can update state.
    fn record_issued(&mut self, warp_idx: usize);

    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Loose Round-Robin (LRR)
// ---------------------------------------------------------------------------

/// Rotates through all warps in order, giving equal priority to each.
/// Simple and fair, but all warps tend to hit long-latency stalls together,
/// leading to the "stall cliff" where the scheduler finds no eligible warp.
pub struct LrrScheduler {
    last_issued: usize,
}

impl LrrScheduler {
    pub fn new() -> Self {
        LrrScheduler { last_issued: 0 }
    }
}

impl WarpScheduler for LrrScheduler {
    fn order_warps(&mut self, slots: &[WarpSlot]) -> Vec<usize> {
        let n = slots.len();
        if n == 0 {
            return vec![];
        }
        // Find the position of last_issued in slots, then rotate from there
        let start_pos = slots
            .iter()
            .position(|s| s.warp_idx == self.last_issued)
            .unwrap_or(0);
        (1..=n)
            .map(|i| slots[(start_pos + i) % n].warp_idx)
            .collect()
    }

    fn record_issued(&mut self, warp_idx: usize) {
        self.last_issued = warp_idx;
    }

    fn name(&self) -> &'static str {
        "LRR"
    }
}

// ---------------------------------------------------------------------------
// Greedy-Then-Oldest (GTO)
// ---------------------------------------------------------------------------

/// Sticks with the same warp until it stalls (greedy), then falls back to
/// the oldest eligible warp. Reduces cache thrashing vs. LRR by serializing
/// each warp's working set rather than interleaving all warps simultaneously.
/// This is the default policy in GPGPU-Sim.
pub struct GtoScheduler {
    last_issued: Option<usize>,
}

impl GtoScheduler {
    pub fn new() -> Self {
        GtoScheduler { last_issued: None }
    }
}

impl WarpScheduler for GtoScheduler {
    fn order_warps(&mut self, slots: &[WarpSlot]) -> Vec<usize> {
        // Greedy warp first (last issued, if still present)
        let mut ordered: Vec<usize> = Vec::with_capacity(slots.len());

        if let Some(last) = self.last_issued {
            if slots.iter().any(|s| s.warp_idx == last) {
                ordered.push(last);
            }
        }

        // Remaining warps sorted by age ascending (oldest = smallest age)
        let mut rest: Vec<&WarpSlot> = slots
            .iter()
            .filter(|s| Some(s.warp_idx) != self.last_issued)
            .collect();
        rest.sort_by_key(|s| s.age);
        ordered.extend(rest.iter().map(|s| s.warp_idx));

        ordered
    }

    fn record_issued(&mut self, warp_idx: usize) {
        self.last_issued = Some(warp_idx);
    }

    fn name(&self) -> &'static str {
        "GTO"
    }
}

// ---------------------------------------------------------------------------
// Two-Level Active Warp Scheduler (TLWS)
// ---------------------------------------------------------------------------

/// Maintains a small active set of warps (scheduled LRR within); the rest sit
/// in a pending pool. When the entire active set stalls, warps are promoted
/// from the pending pool. The active set size is chosen so its combined working
/// set fits in L1 cache, hiding latency without thrashing it.
///
/// ~19% average speedup over LRR across GPU benchmarks (Narasiman et al., MICRO 2011).
pub struct TwoLevelScheduler {
    active_set_size: usize,
    /// Warp indices currently in the active set, in insertion order
    active_set: Vec<usize>,
    /// LRR cursor within the active set
    last_issued_pos: usize,
    /// Global warp issue counter (for age tracking within active set)
    issue_count: u64,
}

impl TwoLevelScheduler {
    pub fn new(active_set_size: usize) -> Self {
        TwoLevelScheduler {
            active_set_size,
            active_set: Vec::new(),
            last_issued_pos: 0,
            issue_count: 0,
        }
    }
}

impl WarpScheduler for TwoLevelScheduler {
    fn order_warps(&mut self, slots: &[WarpSlot]) -> Vec<usize> {
        // Promote warps from pending into active set to fill free slots
        let active_set: std::collections::HashSet<usize> =
            self.active_set.iter().cloned().collect();

        let pending: Vec<usize> = slots
            .iter()
            .filter(|s| !active_set.contains(&s.warp_idx))
            .map(|s| s.warp_idx)
            .collect();

        for idx in pending {
            if self.active_set.len() >= self.active_set_size {
                break;
            }
            self.active_set.push(idx);
        }

        // Build priority list: active set (LRR order) first, then pending
        let active_now: std::collections::HashSet<usize> =
            self.active_set.iter().cloned().collect();

        let n = self.active_set.len();
        let mut ordered: Vec<usize> = if n > 0 {
            let start = self.last_issued_pos % n;
            (1..=n)
                .map(|i| self.active_set[(start + i) % n])
                .collect()
        } else {
            vec![]
        };

        // Pending warps as fallback
        let pending_ordered: Vec<usize> = slots
            .iter()
            .filter(|s| !active_now.contains(&s.warp_idx))
            .map(|s| s.warp_idx)
            .collect();
        ordered.extend(pending_ordered);

        ordered
    }

    fn record_issued(&mut self, warp_idx: usize) {
        if let Some(pos) = self.active_set.iter().position(|&i| i == warp_idx) {
            self.last_issued_pos = pos;
        }
        self.issue_count += 1;
    }

    fn name(&self) -> &'static str {
        "TwoLevel"
    }
}

// ---------------------------------------------------------------------------
// Policy selector
// ---------------------------------------------------------------------------

/// Selectable warp scheduling policy.
pub enum SchedulingPolicy {
    /// Loose Round-Robin
    Lrr,
    /// Greedy-Then-Oldest (default in GPGPU-Sim)
    Gto,
    /// Two-Level Active Warp Scheduling
    TwoLevel {
        /// Number of warps in the active set (typically 4–8)
        active_set_size: usize,
    },
}

impl SchedulingPolicy {
    pub fn build(self) -> Box<dyn WarpScheduler> {
        match self {
            SchedulingPolicy::Lrr => Box::new(LrrScheduler::new()),
            SchedulingPolicy::Gto => Box::new(GtoScheduler::new()),
            SchedulingPolicy::TwoLevel { active_set_size } => {
                Box::new(TwoLevelScheduler::new(active_set_size))
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            SchedulingPolicy::Lrr => "LRR",
            SchedulingPolicy::Gto => "GTO",
            SchedulingPolicy::TwoLevel { .. } => "TwoLevel",
        }
    }
}
