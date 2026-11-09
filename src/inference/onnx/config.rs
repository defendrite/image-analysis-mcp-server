#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionDevice {
    Auto,
    Cpu,
    // Cuda,
}

#[derive(Debug, Clone)]
pub struct OnnxConfig {
    pub input_name: String,
    pub output_name: Option<String>,
    pub device: ExecutionDevice,
    pub intra_threads: usize,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            input_name: "input".to_string(),
            output_name: None,
            device: ExecutionDevice::Auto,
            intra_threads: 4,
        }
    }
}
