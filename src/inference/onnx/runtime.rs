use ort::{environment::Environment, session::Session, value::Tensor as OrtTensor};

use std::{path::Path, sync::Arc};

use crate::inference::engine::backend::{InferenceBackend, Tensor};

use crate::inference::image::ProcessedImage;
use crate::inference::onnx::{
    config::{ExecutionDevice, OnnxConfig},
    model::OnnxModel,
};
use crate::tools::classification::backend::ClassificationBackend;

// ==========================================
// ONNX MODEL LOADING
// ==========================================

impl OnnxModel {
    pub fn load(model_path: impl AsRef<Path>, config: OnnxConfig) -> Result<Self, String> {
        let model_path = model_path.as_ref();

        // Validate model path.

        if !model_path.is_file() {
            return Err(format!(
                "ONNX model does not exist: {}",
                model_path.display()
            ));
        }

        // Validate configuration.

        if config.input_name.trim().is_empty() {
            return Err("ONNX input name cannot be empty".to_string());
        }

        if config
            .output_name
            .as_ref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return Err("ONNX output name cannot be empty".to_string());
        }

        if config.intra_threads == 0 {
            return Err("Thread count must be greater than zero".to_string());
        }

        // Initialize ONNX environment.

        let environment = Arc::new(
            ort::init()
                .build()
                .map_err(|e| format!("Failed to create ONNX environment: {}", e))?,
        );

        // Load execution session.

        let (session, actual_device) = match config.device {
            ExecutionDevice::Cpu => (
                Self::load_cpu(model_path, &config, &environment)?,
                ExecutionDevice::Cpu,
            ),

            // ExecutionDevice::Cuda => (
            //     Self::load_cuda(model_path, &config, &environment)?,
            //     ExecutionDevice::Cuda,
            // ),
            ExecutionDevice::Auto => {
                #[cfg(feature = "cuda")]
                {
                    match Self::load_cuda(model_path, &config, &environment) {
                        Ok(session) => (session, ExecutionDevice::Cuda),

                        Err(error) => {
                            eprintln!("CUDA initialization failed: {}", error);

                            eprintln!("Falling back to CPU execution.");

                            (
                                Self::load_cpu(model_path, &config, &environment)?,
                                ExecutionDevice::Cpu,
                            )
                        }
                    }
                }

                #[cfg(not(feature = "cuda"))]
                {
                    (
                        Self::load_cpu(model_path, &config, &environment)?,
                        ExecutionDevice::Cpu,
                    )
                }
            }
        };

        // Validate input name.

        let input_exists = session
            .inputs()
            .iter()
            .any(|input| input.name() == config.input_name.as_str());

        if !input_exists {
            let available: Vec<String> = session
                .inputs()
                .iter()
                .map(|input| input.name().to_string())
                .collect();

            return Err(format!(
                "ONNX input '{}' not found. Available inputs: {:?}",
                config.input_name, available
            ));
        }

        // Resolve output name.

        let available_outputs: Vec<String> = session
            .outputs()
            .iter()
            .map(|output| output.name().to_string())
            .collect();

        if available_outputs.is_empty() {
            return Err("ONNX model does not contain any outputs".to_string());
        }

        let output_name = match config.output_name {
            Some(name) => name,
            None => available_outputs[0].clone(),
        };

        if !available_outputs.contains(&output_name) {
            return Err(format!(
                "ONNX output '{}' not found. Available outputs: {:?}",
                output_name, available_outputs
            ));
        }

        println!("ONNX model loaded successfully.");
        println!("Input tensor: {}", config.input_name);
        println!("Output tensor: {}", output_name);
        println!("Execution device: {:?}", actual_device);

        Ok(Self {
            _environment: environment,
            session: std::sync::Mutex::new(session),
            input_name: config.input_name,
            output_name,
            // device: actual_device,
        })
    }

    // ======================================
    // CPU SESSION
    // ======================================

    fn load_cpu(
        model_path: &Path,
        config: &OnnxConfig,
        environment: &Environment,
    ) -> Result<Session, String> {
        Session::builder(environment)
            .map_err(|e| format!("CPU session builder failed: {}", e))?
            .with_intra_threads(config.intra_threads)
            .map_err(|e| format!("CPU thread configuration failed: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load ONNX model on CPU: {}", e))
    }

    // ======================================
    // CUDA SESSION
    // ======================================

    // fn load_cuda(
    //     model_path: &Path,
    //     config: &OnnxConfig,
    //     environment: &Environment,
    // ) -> Result<Session, String> {
    //     #[cfg(feature = "cuda")]
    //     {
    //         use ort::ep;

    //         Session::builder(environment)
    //             .map_err(|e| format!("CUDA session builder failed: {}", e))?
    //             .with_intra_threads(config.intra_threads)
    //             .map_err(|e| format!("CUDA thread configuration failed: {}", e))?
    //             .with_execution_providers([ep::CUDA::default().build()])
    //             .map_err(|e| format!("Failed to register CUDA provider: {}", e))?
    //             .commit_from_file(model_path)
    //             .map_err(|e| format!("CUDA model loading failed: {}", e))
    //     }

    //     #[cfg(not(feature = "cuda"))]
    //     {
    //         let _ = (model_path, config, environment);

    //         Err("CUDA support is not enabled".to_string())
    //     }
    // }
}

// ==========================================
// GENERIC INFERENCE IMPLEMENTATION
// ==========================================

impl InferenceBackend for OnnxModel {
    fn infer(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        if inputs.len() != 1 {
            return Err(format!(
                "Expected exactly one input tensor, received {}",
                inputs.len()
            ));
        }

        let input_data = inputs
            .into_iter()
            .next()
            .ok_or_else(|| "Missing input tensor".to_string())?;

        // Create ONNX input tensor.

        let input = OrtTensor::from_array((input_data.shape, input_data.data))
            .map_err(|e| format!("Failed to create ONNX input tensor: {}", e))?;

        // Lock session.

        let mut session = self
            .session
            .lock()
            .map_err(|_| "ONNX session mutex poisoned".to_string())?;

        // Run inference.

        let outputs = session
            .run(ort::inputs! {
                self.input_name.as_str() => input
            })
            .map_err(|e| format!("ONNX inference failed: {}", e))?;

        // Retrieve selected output.

        let output = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| format!("ONNX model returned no output named '{}'", self.output_name))?;

        // Extract tensor.

        let (output_shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract ONNX output as f32: {}", e))?;

        if values.is_empty() {
            return Err("ONNX model returned empty output".to_string());
        }

        let shape: Result<Vec<usize>, String> = output_shape
            .iter()
            .map(|&dim| {
                usize::try_from(dim).map_err(|_| format!("Invalid ONNX output dimension: {}", dim))
            })
            .collect();

        let output_tensor = Tensor::new(shape?, values.to_vec())?;

        println!(
            "ONNX inference completed. Output shape: {:?}",
            output_tensor.shape
        );

        Ok(vec![output_tensor])
    }
}

impl ClassificationBackend for OnnxModel {
    fn predict(&self, image: &ProcessedImage) -> Result<Vec<f32>, String> {
        let input = Tensor::new(image.shape.to_vec(), image.data.clone())?;

        let outputs = InferenceBackend::infer(self, vec![input])?;

        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| "ONNX model returned no output".to_string())?;

        Ok(output.data)
    }
}
