use super::{
    backend::ClassificationBackend,
    types::{ClassificationResult, ClassifierConfig, Prediction},
};
use crate::inference::image::{ProcessedImage, processor::ImageProcessor};

// ==========================================
// CLASSIFIER
// ==========================================

pub struct Classifier<B: ClassificationBackend> {
    backend: B,
    processor: ImageProcessor,
    config: ClassifierConfig,
}

// ==========================================
// CLASSIFIER IMPLEMENTATION
// ==========================================

impl<B: ClassificationBackend> Classifier<B> {
    // ======================================
    // CREATE CLASSIFIER
    // ======================================

    pub fn new(
        backend: B,
        processor: ImageProcessor,
        config: ClassifierConfig,
    ) -> Result<Self, String> {
        if config.top_k == 0 {
            return Err("top_k must be greater than zero".to_string());
        }

        if !config.confidence_threshold.is_finite()
            || !(0.0..=1.0).contains(&config.confidence_threshold)
        {
            return Err("Confidence threshold must be between 0.0 and 1.0".to_string());
        }

        if config.labels.iter().any(|label| label.trim().is_empty()) {
            return Err("Classification labels cannot be empty".to_string());
        }

        Ok(Self {
            backend,
            processor,
            config,
        })
    }

    // ======================================
    // CLASSIFY IMAGE FROM BYTES
    // ======================================

    pub fn classify_bytes(&self, image_bytes: &[u8]) -> Result<ClassificationResult, String> {
        if image_bytes.is_empty() {
            return Err("Input image bytes cannot be empty".to_string());
        }

        let processed_image = self
            .processor
            .process_bytes(image_bytes)
            .map_err(|e| format!("Image preprocessing failed: {}", e))?;

        self.classify_processed(&processed_image)
    }

    // ======================================
    // CLASSIFY IMAGE FROM FILE
    // ======================================

    // pub fn classify_file(
    //     &self,
    //     image_path: impl AsRef<Path>,
    // ) -> Result<ClassificationResult, String> {
    //     let image_path = image_path.as_ref();

    //     if !image_path.is_file() {
    //         return Err(format!(
    //             "Image file does not exist: {}",
    //             image_path.display()
    //         ));
    //     }

    //     let processed_image = self
    //         .processor
    //         .process_file(image_path)
    //         .map_err(|e| format!("Image preprocessing failed: {}", e))?;

    //     self.classify_processed(&processed_image)
    // }

    // ======================================
    // CLASSIFY PREPROCESSED IMAGE
    // ======================================

    pub fn classify_processed(
        &self,
        image: &ProcessedImage,
    ) -> Result<ClassificationResult, String> {
        // Run model inference.
        let logits = self
            .backend
            .predict(image)
            .map_err(|e| format!("Classification inference failed: {}", e))?;

        // Validate model output.
        if logits.is_empty() {
            return Err("Classification model returned empty output".to_string());
        }

        if logits.iter().any(|value| !value.is_finite()) {
            return Err("Classification model returned invalid logits".to_string());
        }

        // Validate labels.
        if !self.config.labels.is_empty() && self.config.labels.len() != logits.len() {
            return Err(format!(
                "Number of labels ({}) does not match model output classes ({})",
                self.config.labels.len(),
                logits.len()
            ));
        }

        // Calculate probabilities.
        let probabilities = softmax(&logits)?;

        // Build predictions.
        let mut predictions: Vec<Prediction> = probabilities
            .into_iter()
            .enumerate()
            .map(|(index, probability)| {
                let label = self
                    .config
                    .labels
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", index));

                Prediction {
                    class_index: index,
                    label,
                    probability,
                }
            })
            .filter(|prediction| prediction.probability >= self.config.confidence_threshold)
            .collect();

        // Sort by probability, highest first.
        predictions.sort_by(|a, b| b.probability.total_cmp(&a.probability));

        // Keep only top K predictions.
        predictions.truncate(self.config.top_k);

        Ok(ClassificationResult { predictions })
    }

    // ======================================
    // ACCESS CLASSIFIER CONFIGURATION
    // ======================================

    // pub fn config(&self) -> &ClassifierConfig {
    //     &self.config
    // }
}

// ==========================================
// SOFTMAX
// ==========================================

fn softmax(logits: &[f32]) -> Result<Vec<f32>, String> {
    if logits.is_empty() {
        return Err("Cannot calculate softmax for empty logits".to_string());
    }

    if logits.iter().any(|value| !value.is_finite()) {
        return Err("Softmax received invalid logits".to_string());
    }

    // Numerical stability.
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Calculate exponentials.
    let exponentials: Vec<f32> = logits
        .iter()
        .map(|value| (*value - max_logit).exp())
        .collect();

    // Calculate normalization factor.
    let sum: f32 = exponentials.iter().sum();

    if !sum.is_finite() || sum <= 0.0 {
        return Err("Unable to calculate classification probabilities".to_string());
    }

    // Normalize probabilities.
    let probabilities: Vec<f32> = exponentials.into_iter().map(|value| value / sum).collect();

    if probabilities.iter().any(|value| !value.is_finite()) {
        return Err("Softmax generated invalid probabilities".to_string());
    }

    Ok(probabilities)
}
