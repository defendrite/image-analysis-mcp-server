use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ==========================================
// TOOL INPUT
// ==========================================

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ClassificationParams {
    /// Base64-encoded image bytes without data URL prefix.
    pub image_base64: String,
}

// ==========================================
// CLASSIFICATION CONFIGURATION
// ==========================================

#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    pub labels: Vec<String>,
    pub top_k: usize,
    pub confidence_threshold: f32,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            labels: Vec::new(),
            top_k: 3,
            confidence_threshold: 0.0,
        }
    }
}

// ==========================================
// CLASSIFICATION PREDICTION
// ==========================================

#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    pub class_index: usize,
    pub label: String,
    pub probability: f32,
}

// ==========================================
// CLASSIFICATION RESULT
// ==========================================

#[derive(Debug, Clone, Serialize)]
pub struct ClassificationResult {
    pub predictions: Vec<Prediction>,
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
