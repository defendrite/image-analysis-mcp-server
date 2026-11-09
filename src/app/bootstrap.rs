use crate::inference::onnx::{OnnxConfig, OnnxModel};
use crate::models::model_loader::ModelLoader;

use crate::models::{
    model_configs::classification_config,
    model_manager::{DownloadedModel, ModelManager},
};

use crate::tools::classification::backend::ClassificationBackend;

use std::path::Path;
use std::sync::Arc;

// ==========================================
// FIND ONNX MODEL FILE
// ==========================================

fn get_onnx_path(model: &DownloadedModel) -> Result<&Path, String> {
    let onnx_file = model
        .files
        .iter()
        .find(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
        })
        .ok_or_else(|| format!("No ONNX file found in downloaded model: {}", model.name))?;

    if !onnx_file.is_file() {
        return Err(format!(
            "ONNX model file does not exist: {}",
            onnx_file.display()
        ));
    }

    Ok(onnx_file.as_path())
}

// ==========================================
// LOAD CLASSIFICATION BACKEND
// ==========================================

fn load_classification_backend(model: &DownloadedModel) -> Result<OnnxModel, String> {
    let model_path = get_onnx_path(model)?;

    println!("Loading classification model: {}", model_path.display());

    let config = OnnxConfig {
        input_name: "input".to_string(),
        output_name: None,
        device: crate::inference::onnx::ExecutionDevice::Auto,
        intra_threads: 4,
    };

    OnnxModel::load(model_path, config)
}

// ==========================================
// APPLICATION BOOTSTRAP
// ==========================================

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Image Analysis MCP Server...");

    // 1. Initialize model manager

    let manager = ModelManager::new()?;

    // 2. Load model configuration

    let classification_config = classification_config();

    // 3. Download classification model

    println!("Preparing classification model...");

    let downloaded_model = manager.download_model(&classification_config).await?;

    println!("Classification model downloaded.");

    // 4. Load model metadata

    let loaded_model = ModelLoader::new().load(downloaded_model.clone())?;

    // 5. Initialize classification backend

    println!("Loading classification ONNX model...");

    let classification_backend = load_classification_backend(&downloaded_model)?;

    let classification_backend: Arc<dyn ClassificationBackend> = Arc::new(classification_backend);

    println!("Classification backend ready.");

    // 6. Extract classification labels

    let classification_labels = loaded_model.metadata.labels.clone();

    // 7. Start MCP server

    println!("Starting MCP server...");

    crate::mcp::server::run(Some(classification_backend), classification_labels).await?;

    println!("MCP server stopped.");

    Ok(())
}
