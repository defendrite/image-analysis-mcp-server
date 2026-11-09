use super::model_config::{DevicePreference, ModelConfig, ModelFormat};

pub fn classification_config() -> ModelConfig {
    ModelConfig::new(
        "image-classification",
        "defendrite/binary-classifier",
        ModelFormat::ONNX,
        vec!["model.onnx".to_string(), "config.json".to_string()],
    )
    .with_revision("main")
    .with_device(DevicePreference::Auto)
}

// pub fn age_detection_config() -> ModelConfig {
//     ModelConfig::new(
//         "age-detection",
//         "your-org/your-age-model",
//         ModelFormat::ONNX,
//         vec![
//             "model.onnx".to_string(),
//             "config.json".to_string(),
//         ],
//     )
//     .with_revision("main")
//     .with_device(DevicePreference::Auto)
// }
