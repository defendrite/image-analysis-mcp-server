use crate::models::model_config::ModelConfig;

use reqwest::{Client, Url};
use std::error::Error;
use std::fmt;
use std::{
    path::{Path, PathBuf},
    result::Result,
};
use tokio::{fs, io::AsyncWriteExt};
// ==========================================
// ERROR TYPE
// ==========================================

#[derive(Debug)]
pub enum ModelError {
    Http(reqwest::Error),
    Io(std::io::Error),
    Url(url::ParseError),
    Validation(String),
    InvalidPath,
    InvalidFilename,
    InvalidUrl,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Http(e) => write!(f, "HTTP error: {}", e),

            ModelError::Io(e) => write!(f, "IO error: {}", e),

            ModelError::Url(e) => write!(f, "URL error: {}", e),

            ModelError::Validation(e) => {
                write!(f, "Model validation error: {}", e)
            }

            ModelError::InvalidPath => {
                write!(f, "Invalid model path")
            }

            ModelError::InvalidFilename => {
                write!(f, "Invalid model filename")
            }

            ModelError::InvalidUrl => {
                write!(f, "Invalid model download URL")
            }
        }
    }
}

impl Error for ModelError {}

impl From<reqwest::Error> for ModelError {
    fn from(error: reqwest::Error) -> Self {
        Self::Http(error)
    }
}

impl From<std::io::Error> for ModelError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<url::ParseError> for ModelError {
    fn from(error: url::ParseError) -> Self {
        Self::Url(error)
    }
}

// ==========================================
// DOWNLOADED MODEL
// ==========================================

#[derive(Debug, Clone)]
pub struct DownloadedModel {
    pub name: String,
    pub repo_id: String,
    // pub format: ModelFormat,
    pub files: Vec<PathBuf>,
    // pub cache_dir: PathBuf,
}

// ==========================================
// MODEL MANAGER
// ==========================================

pub struct ModelManager {
    client: Client,
    default_cache_dir: PathBuf,
}

impl ModelManager {
    pub fn new() -> Result<Self, ModelError> {
        let client = Client::builder()
            .user_agent("image-analysis-mcp-server/0.1.0")
            .build()?;

        let default_cache_dir = PathBuf::from("models_cache");

        Ok(Self {
            client,
            default_cache_dir,
        })
    }

    // ======================================
    // DOWNLOAD COMPLETE MODEL
    // ======================================

    pub async fn download_model(
        &self,
        config: &ModelConfig,
    ) -> Result<DownloadedModel, ModelError> {
        config
            .validate()
            .map_err(|e| ModelError::Validation(e.to_string()))?;

        let cache_root = config
            .cache_dir
            .clone()
            .unwrap_or_else(|| self.default_cache_dir.clone());

        let repo_cache_name = config.repo_id.replace('/', "--");
        let revision_cache_name = config.revision.replace('/', "--");

        let model_cache_dir = cache_root
            .join("huggingface")
            .join(repo_cache_name)
            .join(revision_cache_name);

        fs::create_dir_all(&model_cache_dir).await?;

        let token = std::env::var("HF_TOKEN").ok();

        let mut downloaded_files = Vec::new();

        for filename in &config.files {
            println!("Checking model file: {} / {}", config.repo_id, filename);

            let local_path = model_cache_dir.join(filename);

            if fs::try_exists(&local_path).await? {
                println!("Using cached model file: {}", local_path.display());

                downloaded_files.push(local_path);
                continue;
            }

            self.download_file(
                &config.repo_id,
                &config.revision,
                filename,
                &model_cache_dir,
                token.as_deref(),
            )
            .await?;

            downloaded_files.push(local_path);
        }

        println!("Model {} is available in cache", config.name);

        Ok(DownloadedModel {
            name: config.name.clone(),
            repo_id: config.repo_id.clone(),
            // format: config.format.clone(),
            files: downloaded_files,
            // cache_dir: model_cache_dir,
        })
    }

    // ======================================
    // DOWNLOAD SINGLE FILE
    // ======================================

    async fn download_file(
        &self,
        repo_id: &str,
        revision: &str,
        filename: &str,
        cache_dir: &Path,
        token: Option<&str>,
    ) -> Result<(), ModelError> {
        let url = build_huggingface_url(repo_id, revision, filename)?;

        println!("Downloading: {}", url);

        let mut request = self.client.get(url);

        if let Some(token) = token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?.error_for_status()?;

        let final_path = cache_dir.join(filename);

        let parent = final_path.parent().ok_or(ModelError::InvalidPath)?;

        fs::create_dir_all(parent).await?;

        // Temporary file prevents incomplete downloads
        // from being treated as valid cached models.
        let temp_path = final_path.with_extension("download");

        let download_result: Result<(), ModelError> = async {
            let mut file = fs::File::create(&temp_path).await?;

            let mut response = response;

            while let Some(chunk) = response.chunk().await? {
                file.write_all(&chunk).await?;
            }

            file.flush().await?;
            file.sync_all().await?;

            drop(file);

            fs::rename(&temp_path, &final_path).await?;

            Ok(())
        }
        .await;

        if download_result.is_err() {
            let _ = fs::remove_file(&temp_path).await;
        }

        download_result?;

        println!("Downloaded successfully: {}", final_path.display());

        Ok(())
    }
}

// ==========================================
// HUGGING FACE URL BUILDER
// ==========================================

fn build_huggingface_url(repo_id: &str, revision: &str, filename: &str) -> Result<Url, ModelError> {
    let mut url = Url::parse("https://huggingface.co/")?;

    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| ModelError::InvalidUrl)?;

        segments.pop_if_empty();

        for part in repo_id.split('/') {
            segments.push(part);
        }

        segments.push("resolve");

        for part in revision.split('/') {
            segments.push(part);
        }

        for part in Path::new(filename).components() {
            if let std::path::Component::Normal(value) = part {
                let value = value.to_str().ok_or(ModelError::InvalidFilename)?;

                segments.push(value);
            } else {
                return Err(ModelError::InvalidFilename);
            }
        }
    }

    Ok(url)
}
