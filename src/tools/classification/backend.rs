use std::sync::Arc;

use crate::inference::image::ProcessedImage;

pub trait ClassificationBackend: Send + Sync {
    fn predict(&self, image: &ProcessedImage) -> Result<Vec<f32>, String>;
}

impl<T: ClassificationBackend + ?Sized> ClassificationBackend for Arc<T> {
    fn predict(&self, image: &ProcessedImage) -> Result<Vec<f32>, String> {
        (**self).predict(image)
    }
}
