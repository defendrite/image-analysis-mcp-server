use rmcp::{handler::server::wrapper::Parameters, tool, tool_router};

use std::sync::Arc;

use crate::tools::classification::backend::ClassificationBackend;
use crate::tools::classification::{ClassificationParams, ClassificationTool};

#[derive(Clone)]
pub struct ImageAnalysisServer {
    classification_backend: Option<Arc<dyn ClassificationBackend>>,
    classification_labels: Vec<String>,
}

impl ImageAnalysisServer {
    pub fn new() -> Self {
        Self {
            classification_backend: None,
            classification_labels: Vec::new(),
        }
    }

    pub fn with_classification_backend(
        mut self,
        backend: Arc<dyn ClassificationBackend>,
        labels: Vec<String>,
    ) -> Self {
        self.classification_backend = Some(backend);
        self.classification_labels = labels;

        self
    }
}

#[tool_router(server_handler)]
impl ImageAnalysisServer {
    #[tool(
        description = "Classify an image as either a dog or a cat using a binary classification model. Accepts a Base64-encoded JPEG, PNG, or WebP image and returns the predicted label ('dog' or 'cat') with its confidence score."
    )]
    fn classify_dog_or_cat(
        &self,
        Parameters(params): Parameters<ClassificationParams>,
    ) -> Result<String, String> {
        let response = ClassificationTool::classify(
            &params.image_base64,
            self.classification_backend.as_ref(),
            &self.classification_labels,
        )?;

        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }
}
