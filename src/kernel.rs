/// Kernel definitions and launch configuration.
/// A kernel is a function that every thread executes, identified by its
/// thread/block coordinates — mirroring the CUDA execution model.
use crate::memory::HBM;

/// 3D dimension struct used for grid and block sizes (mirrors CUDA's dim3).
#[derive(Debug, Clone, Copy)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Dim3 { x, y, z }
    }

    /// Convenience constructor for 1D configs
    pub fn x(x: u32) -> Self {
        Dim3 { x, y: 1, z: 1 }
    }

    /// Total number of threads/blocks in this dimension
    pub fn size(&self) -> u32 {
        self.x * self.y * self.z
    }
}

/// Configuration for launching a kernel: how many blocks (grid) and
/// how many threads per block (block).
pub struct LaunchConfig {
    pub grid_dim: Dim3,
    pub block_dim: Dim3,
}

impl LaunchConfig {
    pub fn new(grid_dim: Dim3, block_dim: Dim3) -> Self {
        LaunchConfig { grid_dim, block_dim }
    }

    /// Total number of thread blocks in the grid
    pub fn num_blocks(&self) -> u32 {
        self.grid_dim.size()
    }

    /// Total number of threads per block
    pub fn threads_per_block(&self) -> u32 {
        self.block_dim.size()
    }
}

/// Per-thread context passed into the kernel function.
/// Contains thread/block coordinates and access to shared + global memory.
pub struct ThreadCtx<'a> {
    pub thread_idx: Dim3,
    pub block_idx: Dim3,
    pub block_dim: Dim3,
    pub grid_dim: Dim3,
    /// Per-block shared memory (SMEM) — shared among all threads in the block
    pub smem: &'a mut Vec<u8>,
    /// Global memory (HBM)
    pub gmem: &'a mut HBM,
}

impl<'a> ThreadCtx<'a> {
    /// Flat 1D global thread index: blockIdx.x * blockDim.x + threadIdx.x
    pub fn global_id(&self) -> u32 {
        self.block_idx.x * self.block_dim.x + self.thread_idx.x
    }
}

/// A GPU kernel: a named function executed by every thread in the launch grid.
pub struct Kernel {
    pub name: String,
    pub func: Box<dyn Fn(&mut ThreadCtx<'_>)>,
}

impl Kernel {
    pub fn new<F>(name: &str, func: F) -> Self
    where
        F: Fn(&mut ThreadCtx<'_>) + 'static,
    {
        Kernel {
            name: name.to_string(),
            func: Box::new(func),
        }
    }
}
