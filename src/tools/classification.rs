
use base64::{
    engine::general_purpose::STANDARD,
    Engine as _,
};

use rmcp::{
    handler::server::wrapper::Parameters,
    tool,
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::processors::{
    classifier::{
        ClassificationBackend,
        ClassificationResult,
        Classifier,
        ClassifierConfig,
    },
    image_processor::{
        ImageProcessor,
        ImageProcessorConfig,
    },
};

// ==========================================
// TOOL INPUT
// ==========================================

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ClassificationParams {
    /// Base64-encoded image bytes (without data URL prefix).
    pub image_base64: String,
}

// ==========================================
// TOOL RESPONSE
// ==========================================

#[derive(Debug, Serialize)]
pub struct ClassificationResponse {
    pub success: bool,
    pub predictions: Option<ClassificationResult>,
    pub error: Option<String>,
}

// ==========================================
// CLASSIFICATION TOOL
// ==========================================

pub struct ClassificationTool;

impl ClassificationTool {

    pub fn classify(
        image_base64: &str,
        backend: Option<&Arc<dyn ClassificationBackend>>,
        labels: &[String],
    ) -> Result<ClassificationResponse, String> {

        // Check input size before decoding.
        const MAX_BASE64_SIZE: usize = 20 * 1024 * 1024;

        if image_base64.len() > MAX_BASE64_SIZE {
            return Err("Image input exceeds the 20 MB Base64 limit".to_string());
        }

        // Check whether an inference backend is available.
        let backend = backend.ok_or_else(|| {
            "Classification inference backend is not configured yet".to_string()
        })?;

        // Decode Base64 image.
        let image_bytes = STANDARD
            .decode(image_base64)
            .map_err(|_| "Invalid Base64 image input".to_string())?;

        // Initialize shared image processor.
        let processor = ImageProcessor::new(
            ImageProcessorConfig::default()
        )
        .map_err(|e| e.to_string())?;

        // Configure classifier.
        let config = ClassifierConfig {
            labels: labels.to_vec(),
            top_k: 3,
        };

        // Create classifier using the configured backend.
        let classifier = Classifier::new(
            Arc::clone(backend),
            processor,
            config,
        );

        // Run classification.
        let result = classifier
            .classify_bytes(&image_bytes)
            .map_err(|e| e.to_string())?;

        Ok(ClassificationResponse {
            success: true,
            predictions: Some(result),
            error: None,
        })
    }
}
