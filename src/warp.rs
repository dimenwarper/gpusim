/// Warp and WarpScheduler simulation.
/// A warp is a group of 32 threads executing in SIMD lockstep.
/// The WarpScheduler manages warp execution within an SM subpartition.

pub const WARP_SIZE: usize = 32; // Standard CUDA warp size

/// A warp: 32 threads executing the same instruction in SIMD fashion.
pub struct Warp {
    pub id: usize,
    /// Program counter — which instruction the warp is currently executing
    pub pc: usize,
    /// Register file for each thread in the warp
    pub registers: Vec<[u32; 32]>, // 32 registers per thread
    pub active: bool,
    /// Launch timestamp — lower means older. Used by GTO for age-based priority.
    pub age: u64,
}

impl Warp {
    pub fn new(id: usize) -> Self {
        Warp {
            id,
            pc: 0,
            registers: vec![[0u32; 32]; WARP_SIZE],
            active: true,
            age: 0,
        }
    }

    pub fn with_age(mut self, age: u64) -> Self {
        self.age = age;
        self
    }
}

/// Schedules and manages warp execution within an SM subpartition.
pub struct WarpScheduler {
    pub current_warp: Option<Warp>,
}

impl WarpScheduler {
    pub fn new() -> Self {
        WarpScheduler { current_warp: None }
    }

    pub fn is_idle(&self) -> bool {
        self.current_warp.is_none()
    }

    pub fn load_warp(&mut self, warp: Warp) {
        self.current_warp = Some(warp);
    }

    pub fn tick(&mut self) {
        if let Some(ref mut warp) = self.current_warp {
            // Advance the program counter by one instruction per tick
            warp.pc += 1;
        }
    }
}
