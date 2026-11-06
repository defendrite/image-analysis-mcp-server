
use anyhow::{Context, Result, anyhow};
use image::{
    DynamicImage,
    ImageReader,
    imageops::FilterType,
};
use std::path::Path;

// ==========================================
// IMAGE PROCESSOR CONFIGURATION
// ==========================================

#[derive(Debug, Clone)]
pub struct ImageProcessorConfig {
    pub width: u32,
    pub height: u32,

    // RGB normalization values
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            width: 224,
            height: 224,

            // Standard ImageNet normalization
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

// ==========================================
// PROCESSED IMAGE
// ==========================================

#[derive(Debug, Clone)]
pub struct ProcessedImage {
    // NCHW format: [1, 3, height, width]
    pub data: Vec<f32>,

    pub shape: [usize; 4],
}

// ==========================================
// IMAGE PROCESSOR
// ==========================================

pub struct ImageProcessor {
    config: ImageProcessorConfig,
}

impl ImageProcessor {
    pub fn new(config: ImageProcessorConfig) -> Result<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(anyhow!("Image dimensions must be greater than zero"));
        }

        for std in config.std {
            if !std.is_finite() || std <= 0.0 {
                return Err(anyhow!(
                    "Normalization standard deviation must be positive and finite"
                ));
            }
        }

        for value in config.mean {
            if !value.is_finite() {
                return Err(anyhow!(
                    "Normalization mean must be finite"
                ));
            }
        }

        Ok(Self { config })
    }

    // ======================================
    // PROCESS IMAGE FROM BYTES
    // ======================================

    pub fn process_bytes(&self, bytes: &[u8]) -> Result<ProcessedImage> {
        let image = image::load_from_memory(bytes)
            .context("Failed to decode image")?;

        self.process_image(image)
    }

    // ======================================
    // PROCESS IMAGE FROM FILE
    // ======================================

    pub fn process_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessedImage> {
        let image = ImageReader::open(path)?
            .with_guessed_format()?
            .decode()
            .context("Failed to decode image file")?;

        self.process_image(image)
    }

    // ======================================
    // MAIN PREPROCESSING PIPELINE
    // ======================================

    fn process_image(
        &self,
        image: DynamicImage,
    ) -> Result<ProcessedImage> {

        // Resize image to model input dimensions
        let resized = image.resize_exact(
            self.config.width,
            self.config.height,
            FilterType::Triangle,
        );

        // Convert to RGB
        let rgb = resized.to_rgb8();

        let width = self.config.width as usize;
        let height = self.config.height as usize;

        let plane_size = width * height;

        let mut data = vec![0.0_f32; 3 * plane_size];

        // Convert HWC RGB pixels to NCHW
        for (x, y, pixel) in rgb.enumerate_pixels() {

            let x = x as usize;
            let y = y as usize;

            let pixel_index = y * width + x;

            for channel in 0..3 {

                let pixel_value =
                    pixel[channel] as f32 / 255.0;

                let normalized =
                    (pixel_value - self.config.mean[channel])
                    / self.config.std[channel];

                let tensor_index =
                    channel * plane_size + pixel_index;

                data[tensor_index] = normalized;
            }
        }

        Ok(ProcessedImage {
            data,
            shape: [1, 3, height, width],
        })
    }

    // ======================================
    // ACCESS CONFIGURATION
    // ======================================

    pub fn config(&self) -> &ImageProcessorConfig {
        &self.config
    }
}
