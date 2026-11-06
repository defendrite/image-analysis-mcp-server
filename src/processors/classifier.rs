
use anyhow::{Result, anyhow};

use crate::processors::image_processor::{
    ImageProcessor,
    ProcessedImage,
};

// ==========================================
// CLASSIFICATION BACKEND
// ==========================================

/// Every inference engine will implement this trait.
///
/// Future implementations:
/// - ONNX Runtime
/// - PyTorch inference
/// - Custom CNN inference
pub trait ClassificationBackend: Send + Sync {

    /// Accepts a preprocessed image and returns raw logits.
    fn predict(
        &self,
        image: &ProcessedImage,
    ) -> Result<Vec<f32>>;
}

// ==========================================
// CLASSIFICATION CONFIGURATION
// ==========================================

#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    // Names corresponding to model output classes
    pub labels: Vec<String>,

    // Number of top predictions to return
    pub top_k: usize,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            labels: Vec::new(),
            top_k: 3,
        }
    }
}

// ==========================================
// CLASSIFICATION RESULT
// ==========================================

#[derive(Debug, Clone)]
pub struct Prediction {
    pub class_index: usize,
    pub label: String,
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub predictions: Vec<Prediction>,
}

// ==========================================
// CLASSIFIER
// ==========================================

pub struct Classifier<B: ClassificationBackend> {
    backend: B,
    processor: ImageProcessor,
    config: ClassifierConfig,
}

impl<B: ClassificationBackend> Classifier<B> {

    pub fn new(
        backend: B,
        processor: ImageProcessor,
        config: ClassifierConfig,
    ) -> Self {
        Self {
            backend,
            processor,
            config,
        }
    }

    // ======================================
    // CLASSIFY IMAGE FROM BYTES
    // ======================================

    pub fn classify_bytes(
        &self,
        image_bytes: &[u8],
    ) -> Result<ClassificationResult> {

        let processed_image =
            self.processor.process_bytes(image_bytes)?;

        self.classify_processed(&processed_image)
    }

    // ======================================
    // CLASSIFY IMAGE FROM FILE
    // ======================================

    pub fn classify_file(
        &self,
        image_path: impl AsRef<std::path::Path>,
    ) -> Result<ClassificationResult> {

        let processed_image =
            self.processor.process_file(image_path)?;

        self.classify_processed(&processed_image)
    }

    // ======================================
    // CLASSIFY PREPROCESSED IMAGE
    // ======================================

    pub fn classify_processed(
        &self,
        image: &ProcessedImage,
    ) -> Result<ClassificationResult> {

        // Run model inference
        let logits = self.backend.predict(image)?;

        if logits.is_empty() {
            return Err(anyhow!(
                "Classification model returned empty output"
            ));
        }

        if logits.iter().any(|value| !value.is_finite()) {
            return Err(anyhow!(
                "Classification model returned invalid logits"
            ));
        }

        if !self.config.labels.is_empty()
            && self.config.labels.len() != logits.len()
        {
            return Err(anyhow!(
                "Number of labels does not match model output classes"
            ));
        }

        // Convert logits into probabilities
        let probabilities = softmax(&logits)?;

        // Generate predictions
        let mut predictions: Vec<Prediction> = probabilities
            .into_iter()
            .enumerate()
            .map(|(index, probability)| {

                let label = self.config.labels
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| {
                        format!("class_{}", index)
                    });

                Prediction {
                    class_index: index,
                    label,
                    probability,
                }
            })
            .collect();

        // Highest probability first
        predictions.sort_by(|a, b| {
            b.probability
                .total_cmp(&a.probability)
        });

        // Keep only top K predictions
        predictions.truncate(
            self.config.top_k.max(1).min(predictions.len())
        );

        Ok(ClassificationResult {
            predictions,
        })
    }
}

// ==========================================
// SOFTMAX
// ==========================================

fn softmax(logits: &[f32]) -> Result<Vec<f32>> {

    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let exponentials: Vec<f32> = logits
        .iter()
        .map(|value| (*value - max_logit).exp())
        .collect();

    let sum: f32 = exponentials.iter().sum();

    if !sum.is_finite() || sum <= 0.0 {
        return Err(anyhow!(
            "Unable to calculate classification probabilities"
        ));
    }

    Ok(
        exponentials
            .into_iter()
            .map(|value| value / sum)
            .collect()
    )
}
