use base64::{Engine as _, engine::general_purpose::STANDARD};

use std::sync::Arc;

use super::{
    backend::ClassificationBackend,
    classifier::Classifier,
    types::{ClassificationResponse, ClassifierConfig},
};

use crate::inference::image::{ImageProcessor, ImageProcessorConfig};

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
        const MAX_BASE64_SIZE: usize = 20 * 1024 * 1024;

        // Validate input.

        if image_base64.trim().is_empty() {
            return Err("Image Base64 input cannot be empty".to_string());
        }

        if image_base64.len() > MAX_BASE64_SIZE {
            return Err("Image input exceeds the 20 MB Base64 limit".to_string());
        }

        // Validate backend.

        let backend = backend
            .ok_or_else(|| "Classification inference backend is not configured".to_string())?;

        // Decode Base64.

        let image_bytes = STANDARD
            .decode(image_base64.trim())
            .map_err(|e| format!("Invalid Base64 image input: {}", e))?;

        if image_bytes.is_empty() {
            return Err("Decoded image data is empty".to_string());
        }

        // Initialize image processor.

        let processor = ImageProcessor::new(ImageProcessorConfig::default())
            .map_err(|e| format!("Failed to initialize image processor: {}", e))?;

        // Configure classifier.

        let config = ClassifierConfig {
            labels: labels.to_vec(),
            top_k: 3,
            confidence_threshold: 0.0,
        };

        // Create classifier.

        let classifier = Classifier::new(Arc::clone(backend), processor, config)
            .map_err(|e| format!("Failed to initialize classifier: {}", e))?;

        // Run classification.

        let result = classifier
            .classify_bytes(&image_bytes)
            .map_err(|e| format!("Image classification failed: {}", e))?;

        // println!("Predictions {:?}", result);

        Ok(ClassificationResponse {
            success: true,
            predictions: Some(result),
            error: None,
        })
    }

    // ======================================
    // SAFE CLASSIFICATION
    // ======================================

    // pub fn classify_safe(
    //     image_base64: &str,
    //     backend: Option<&Arc<dyn ClassificationBackend>>,
    //     labels: &[String],
    // ) -> ClassificationResponse {
    //     match Self::classify(image_base64, backend, labels) {
    //         Ok(response) => response,

    //         Err(error) => ClassificationResponse {
    //             success: false,
    //             predictions: None,
    //             error: Some(error),
    //         },
    //     }
    // }
}
