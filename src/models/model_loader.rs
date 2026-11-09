use std::fs;

use serde::{Deserialize, Serialize};

use super::model_manager::DownloadedModel;

// ==========================================
// MODEL NORMALIZATION
// ==========================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelNormalization {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

// ==========================================
// MODEL METADATA
// ==========================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelMetadata {
    pub name: String,

    #[serde(default)]
    pub repo_id: String,

    pub format: String,

    pub input_name: String,

    pub input_shape: Vec<usize>,

    pub output_name: String,

    pub output_type: String,

    pub num_classes: usize,

    pub labels: Vec<String>,

    pub image_size: usize,

    pub normalization: ModelNormalization,
}

// ==========================================
// LOADED MODEL
// ==========================================

#[derive(Debug, Clone)]
pub struct LoadedModel {
    // pub name: String,
    // pub repo_id: String,
    // pub format: ModelFormat,

    // pub model_files: Vec<PathBuf>,

    // pub model_path: PathBuf,

    // Loaded config.json metadata.
    pub metadata: ModelMetadata,
    // Path to metadata file.
    // pub metadata_path: PathBuf,
}

// ==========================================
// MODEL LOADER
// ==========================================

pub struct ModelLoader;

impl ModelLoader {
    pub fn new() -> Self {
        Self
    }

    // ======================================
    // LOAD MODEL
    // ======================================

    pub fn load(&self, downloaded_model: DownloadedModel) -> Result<LoadedModel, String> {
        if downloaded_model.files.is_empty() {
            return Err(format!(
                "No files found for model: {}",
                downloaded_model.name
            ));
        }

        // ==================================
        // VERIFY DOWNLOADED FILES
        // ==================================

        for file in &downloaded_model.files {
            if !file.is_file() {
                return Err(format!("Model file does not exist: {}", file.display()));
            }
        }

        // ==================================
        // FIND PRIMARY MODEL FILE
        // ==================================

        let model_path = downloaded_model
            .files
            .iter()
            .find(|file| {
                matches!(
                    file.extension().and_then(|ext| ext.to_str()),
                    Some("onnx" | "pt" | "pth" | "safetensors")
                )
            })
            .cloned()
            .ok_or_else(|| {
                format!(
                    "No supported model file found for: {}",
                    downloaded_model.name
                )
            })?;

        // ==================================
        // FIND CONFIG.JSON
        // ==================================

        let metadata_path = downloaded_model
            .files
            .iter()
            .find(|file| file.file_name().and_then(|name| name.to_str()) == Some("config.json"))
            .cloned()
            .ok_or_else(|| "Model config.json was not downloaded".to_string())?;

        // ==================================
        // READ CONFIG.JSON
        // ==================================

        let config_contents = fs::read_to_string(&metadata_path)
            .map_err(|e| format!("Failed to read model config.json: {}", e))?;

        // ==================================
        // DESERIALIZE METADATA
        // ==================================

        let metadata: ModelMetadata = serde_json::from_str(&config_contents)
            .map_err(|e| format!("Failed to deserialize model metadata: {}", e))?;

        // ==================================
        // VALIDATE METADATA
        // ==================================

        Self::validate_metadata(&metadata)?;

        // ==================================
        // VERIFY MODEL FORMAT
        // ==================================

        if metadata.format.to_lowercase() != "onnx" {
            return Err(format!(
                "Expected ONNX metadata format, received '{}'",
                metadata.format
            ));
        }

        println!("Model loaded: {}", metadata.name);

        println!("Repository: {}", downloaded_model.repo_id);

        println!("Model path: {}", model_path.display());

        println!("Metadata path: {}", metadata_path.display());

        println!("Classification labels: {:?}", metadata.labels);

        println!("Number of classes: {}", metadata.num_classes);

        // ==================================
        // RETURN LOADED MODEL
        // ==================================

        Ok(LoadedModel {
            // name: downloaded_model.name,
            // repo_id: downloaded_model.repo_id,
            // format: downloaded_model.format,
            // model_files: downloaded_model.files,
            // model_path,
            metadata,
            // metadata_path,
        })
    }

    // ======================================
    // VALIDATE METADATA
    // ======================================

    fn validate_metadata(metadata: &ModelMetadata) -> Result<(), String> {
        if metadata.name.trim().is_empty() {
            return Err("Model name cannot be empty".to_string());
        }

        if metadata.input_name.trim().is_empty() {
            return Err("Model input name cannot be empty".to_string());
        }

        if metadata.output_name.trim().is_empty() {
            return Err("Model output name cannot be empty".to_string());
        }

        if metadata.num_classes == 0 {
            return Err("Number of classes must be greater than zero".to_string());
        }

        if metadata.labels.len() != metadata.num_classes {
            return Err(format!(
                "Label count ({}) does not match class count ({})",
                metadata.labels.len(),
                metadata.num_classes
            ));
        }

        if metadata.labels.iter().any(|label| label.trim().is_empty()) {
            return Err("Model contains empty classification labels".to_string());
        }

        if metadata.image_size == 0 {
            return Err("Model image size must be greater than zero".to_string());
        }

        if metadata.input_shape != [1, 3, metadata.image_size, metadata.image_size] {
            return Err(format!(
                "Invalid model input shape: {:?}",
                metadata.input_shape
            ));
        }

        if metadata.normalization.mean.len() != 3 || metadata.normalization.std.len() != 3 {
            return Err("Normalization must contain three mean and std values".to_string());
        }

        if metadata.normalization.mean.iter().any(|v| !v.is_finite())
            || metadata
                .normalization
                .std
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("Invalid model normalization values".to_string());
        }

        if metadata.output_type != "logits" {
            return Err(format!(
                "Unsupported model output type: {}",
                metadata.output_type
            ));
        }

        Ok(())
    }
}
