
use anyhow::{Context, Result, anyhow};
use ort::{
    session::Session,
    value::Tensor,
};

use std::{
    path::Path,
    sync::Mutex,
};

use crate::processors::{
    classifier::ClassificationBackend,
    image_processor::ProcessedImage,
};

// ==========================================
// EXECUTION DEVICE
// ==========================================

#[derive(Debug, Clone, Copy)]
pub enum ExecutionDevice {
    Auto,
    Cpu,
    Cuda,
}

// ==========================================
// ONNX MODEL CONFIGURATION
// ==========================================

#[derive(Debug, Clone)]
pub struct OnnxConfig {
    pub input_name: String,
    pub device: ExecutionDevice,
    pub intra_threads: usize,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            input_name: "input".to_string(),
            device: ExecutionDevice::Auto,
            intra_threads: 4,
        }
    }
}

// ==========================================
// ONNX MODEL
// ==========================================

pub struct OnnxModel {
    session: Mutex<Session>,
    input_name: String,
    device: ExecutionDevice,
}

impl OnnxModel {

    // ======================================
    // LOAD MODEL
    // ======================================

    pub fn load(
        model_path: impl AsRef<Path>,
        config: OnnxConfig,
    ) -> Result<Self> {

        let model_path = model_path.as_ref();

        if !model_path.exists() {
            return Err(anyhow!(
                "ONNX model does not exist: {}",
                model_path.display()
            ));
        }

        if config.input_name.trim().is_empty() {
            return Err(anyhow!("ONNX input name cannot be empty"));
        }

        if config.intra_threads == 0 {
            return Err(anyhow!("Thread count must be greater than zero"));
        }

        // Try GPU first when Auto is selected.
        let (session, actual_device) = match config.device {

            ExecutionDevice::Cpu => {
                (
                    Self::load_cpu(model_path, &config)?,
                    ExecutionDevice::Cpu,
                )
            }

            ExecutionDevice::Cuda => {
                (
                    Self::load_cuda(model_path, &config)?,
                    ExecutionDevice::Cuda,
                )
            }

            ExecutionDevice::Auto => {

                #[cfg(feature = "cuda")]
                {
                    match Self::load_cuda(model_path, &config) {
                        Ok(session) => {
                            println!("ONNX model loaded using CUDA");
                            (session, ExecutionDevice::Cuda)
                        }

                        Err(error) => {
                            eprintln!(
                                "CUDA unavailable or failed: {}",
                                error
                            );

                            println!("Falling back to CPU");

                            (
                                Self::load_cpu(model_path, &config)?,
                                ExecutionDevice::Cpu,
                            )
                        }
                    }
                }

                #[cfg(not(feature = "cuda"))]
                {
                    println!(
                        "CUDA support not compiled. Using CPU."
                    );

                    (
                        Self::load_cpu(model_path, &config)?,
                        ExecutionDevice::Cpu,
                    )
                }
            }
        };

        println!(
            "ONNX model loaded successfully: {}",
            model_path.display()
        );

        Ok(Self {
            session: Mutex::new(session),
            input_name: config.input_name,
            device: actual_device,
        })
    }

    // ======================================
    // LOAD CPU SESSION
    // ======================================

    fn load_cpu(
        model_path: &Path,
        config: &OnnxConfig,
    ) -> Result<Session> {

        let session = Session::builder()?
            .with_intra_threads(config.intra_threads)?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model on CPU")?;

        Ok(session)
    }

    // ======================================
    // LOAD CUDA SESSION
    // ======================================

    fn load_cuda(
        model_path: &Path,
        config: &OnnxConfig,
    ) -> Result<Session> {

        #[cfg(feature = "cuda")]
        {
            use ort::ep;

            let session = Session::builder()?
                .with_intra_threads(config.intra_threads)?
                .with_execution_providers([
                    ep::CUDA::default().build()
                ])?
                .commit_from_file(model_path)
                .context("Failed to load ONNX model with CUDA")?;

            Ok(session)
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (model_path, config);

            Err(anyhow!(
                "CUDA support is not enabled in Cargo.toml"
            ))
        }
    }

    // ======================================
    // RUN INFERENCE
    // ======================================

    pub fn predict(
        &self,
        image: &ProcessedImage,
    ) -> Result<Vec<f32>> {

        let expected_size: usize =
            image.shape.iter().product();

        if image.data.len() != expected_size {
            return Err(anyhow!(
                "Input tensor shape does not match data length"
            ));
        }

        if image.data.iter().any(|value| !value.is_finite()) {
            return Err(anyhow!(
                "Input tensor contains invalid values"
            ));
        }

        let input = Tensor::from_array((
            image.shape.to_vec(),
            image.data.clone(),
        ))?;

        let mut session = self.session
            .lock()
            .map_err(|_| anyhow!("ONNX session mutex poisoned"))?;

        let outputs = session.run(
            ort::inputs! {
                self.input_name.as_str() => input
            }
        )?;

        let output = outputs
            .get(0)
            .ok_or_else(|| anyhow!("ONNX model returned no outputs"))?;

        let (_shape, values) = output
            .try_extract_tensor::<f32>()
            .context("Failed to extract ONNX output as f32 tensor")?;

        Ok(values.to_vec())
    }

    // ======================================
    // DEVICE INFORMATION
    // ======================================

    pub fn device(&self) -> ExecutionDevice {
        self.device
    }
}

// ==========================================
// CLASSIFICATION BACKEND IMPLEMENTATION
// ==========================================

impl ClassificationBackend for OnnxModel {

    fn predict(
        &self,
        image: &ProcessedImage,
    ) -> Result<Vec<f32>> {
        OnnxModel::predict(self, image)
    }
}
