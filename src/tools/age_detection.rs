
use base64::{
    engine::general_purpose::STANDARD,
    Engine as _,
};

use serde::Serialize;
use std::sync::Arc;

use crate::processors::image_processor::{
    ImageProcessor,
    ImageProcessorConfig,
    ProcessedImage,
};

// ==========================================
// AGE ESTIMATION BACKEND
// ==========================================

/// Every age estimation inference engine
/// will implement this trait.
///
/// Future implementations:
/// - ONNX Runtime
/// - PyTorch inference
/// - Custom CNN
pub trait AgeDetectionBackend: Send + Sync {

    /// Receives a preprocessed image and returns
    /// an estimated age and optional confidence.
    fn predict(
        &self,
        image: &ProcessedImage,
    ) -> anyhow::Result<AgeEstimate>;
}

// ==========================================
// AGE ESTIMATION RESULT
// ==========================================

#[derive(Debug, Clone, Serialize)]
pub struct AgeEstimate {

    /// Estimated age in years.
    pub estimated_age_years: f32,

    /// Optional model confidence, between 0.0 and 1.0.
    ///
    /// This is model-specific and should not be
    /// interpreted as a guarantee of accuracy.
    pub confidence: Option<f32>,
}

// ==========================================
// TOOL RESPONSE
// ==========================================

#[derive(Debug, Serialize)]
pub struct AgeDetectionResponse {
    pub success: bool,
    pub result: Option<AgeEstimate>,
    pub error: Option<String>,
}

// ==========================================
// AGE DETECTION TOOL
// ==========================================

pub struct AgeDetectionTool;

impl AgeDetectionTool {

    /// Estimate age from a Base64-encoded image.
    ///
    /// The supplied image should contain a visible face.
    /// Face detection and cropping will be handled by
    /// the future age inference pipeline.
    pub fn detect(
        image_base64: &str,
        backend: Option<&Arc<dyn AgeDetectionBackend>>,
    ) -> Result<AgeDetectionResponse, String> {

        // Limit incoming Base64 data to 20 MB.
        const MAX_BASE64_SIZE: usize = 20 * 1024 * 1024;

        if image_base64.len() > MAX_BASE64_SIZE {
            return Err(
                "Image input exceeds the 20 MB Base64 limit".to_string()
            );
        }

        // Check inference backend.
        let backend = backend.ok_or_else(|| {
            "Age detection inference backend is not configured yet".to_string()
        })?;

        // Decode Base64 image.
        let image_bytes = STANDARD
            .decode(image_base64)
            .map_err(|_| {
                "Invalid Base64 image input".to_string()
            })?;

        // Shared image preprocessing.
        let processor = ImageProcessor::new(
            ImageProcessorConfig::default()
        )
        .map_err(|e| e.to_string())?;

        let processed_image = processor
            .process_bytes(&image_bytes)
            .map_err(|e| e.to_string())?;

        // Run age estimation.
        let estimate = backend
            .predict(&processed_image)
            .map_err(|e| e.to_string())?;

        // Validate model output.
        if !estimate.estimated_age_years.is_finite()
            || !(0.0..=120.0).contains(&estimate.estimated_age_years)
        {
            return Err(
                "Age estimation model returned an invalid age".to_string()
            );
        }

        if let Some(confidence) = estimate.confidence {
            if !confidence.is_finite()
                || !(0.0..=1.0).contains(&confidence)
            {
                return Err(
                    "Age estimation model returned invalid confidence".to_string()
                );
            }
        }

        Ok(AgeDetectionResponse {
            success: true,
            result: Some(estimate),
            error: None,
        })
    }
}
