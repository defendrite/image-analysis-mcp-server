
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ==========================================
// MODEL FORMAT
// ==========================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFormat {
    PyTorch,
    ONNX,
    SafeTensors,
    Custom,
}

// ==========================================
// DEVICE PREFERENCE
// ==========================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DevicePreference {
    Auto,
    CPU,
    CUDA,
}

// ==========================================
// MODEL CONFIGURATION
// ==========================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Unique model identifier used by our application
    pub name: String,

    // Hugging Face repository
    // Example: organization/model-name
    pub repo_id: String,

    // Hugging Face branch, tag or commit
    pub revision: String,

    // Model format
    pub format: ModelFormat,

    // Files to download from Hugging Face
    // Example: model.onnx, config.json
    pub files: Vec<String>,

    // Local model cache directory
    pub cache_dir: Option<PathBuf>,

    // Preferred inference device
    pub device: DevicePreference,
}

impl ModelConfig {
    pub fn new(
        name: impl Into<String>,
        repo_id: impl Into<String>,
        format: ModelFormat,
        files: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            repo_id: repo_id.into(),
            revision: "main".to_string(),
            format,
            files,
            cache_dir: None,
            device: DevicePreference::Auto,
        }
    }

    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }

    pub fn with_cache_dir(mut self, path: PathBuf) -> Self {
        self.cache_dir = Some(path);
        self
    }

    pub fn with_device(mut self, device: DevicePreference) -> Self {
        self.device = device;
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("Model name cannot be empty".to_string());
        }

        // Expect organization/model or user/model
        let repo_parts: Vec<&str> = self.repo_id.split('/').collect();

        if repo_parts.len() != 2
            || repo_parts.iter().any(|part| {
                part.is_empty()
                    || !part.chars().all(|c| {
                        c.is_ascii_alphanumeric()
                            || c == '-'
                            || c == '_'
                            || c == '.'
                    })
                    || *part == "."
                    || *part == ".."
            })
        {
            return Err("Invalid Hugging Face repository ID".to_string());
        }

        if self.revision.trim().is_empty()
            || self.revision.split('/').any(|part| {
                part.is_empty() || part == "." || part == ".."
            })
        {
            return Err("Invalid model revision".to_string());
        }

        if self.files.is_empty() {
            return Err("At least one model file is required".to_string());
        }

        for file in &self.files {
            validate_relative_path(file)?;
        }

        Ok(())
    }
}

// ==========================================
// PATH VALIDATION
// ==========================================

fn validate_relative_path(path: &str) -> Result<(), String> {
    let path = std::path::Path::new(path);

    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err("Invalid model file path".to_string());
    }

    for component in path.components() {
        match component {
            std::path::Component::Normal(_) => {}
            _ => return Err("Invalid model file path".to_string()),
        }
    }

    Ok(())
}
