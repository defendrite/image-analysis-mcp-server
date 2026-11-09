use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFormat {
    ONNX,
    PyTorch,
    SafeTensors,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DevicePreference {
    Auto,
    CPU,
    CUDA,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub repo_id: String,
    pub revision: String,
    pub format: ModelFormat,
    pub files: Vec<String>,
    pub cache_dir: Option<PathBuf>,
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
            revision: "main".into(),
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

    // pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
    //     self.cache_dir = Some(path.into());
    //     self
    // }

    pub fn with_device(mut self, device: DevicePreference) -> Self {
        self.device = device;
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("Model name cannot be empty".into());
        }

        validate_repo_id(&self.repo_id)?;
        validate_revision(&self.revision)?;

        if self.files.is_empty() {
            return Err("At least one model file is required".into());
        }

        for file in &self.files {
            validate_relative_path(file)?;
        }

        Ok(())
    }
}

fn validate_repo_id(repo_id: &str) -> Result<(), String> {
    let parts: Vec<&str> = repo_id.split('/').collect();

    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || !part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
                || *part == "."
                || *part == ".."
        })
    {
        return Err("Invalid Hugging Face repository ID".into());
    }

    Ok(())
}

fn validate_revision(revision: &str) -> Result<(), String> {
    if revision.trim().is_empty()
        || revision
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."))
    {
        return Err("Invalid model revision".into());
    }

    Ok(())
}

fn validate_relative_path(path: &str) -> Result<(), String> {
    let path = Path::new(path);

    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err("Invalid model file path".into());
    }

    Ok(())
}
