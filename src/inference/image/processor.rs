use image::{DynamicImage, imageops::FilterType};

use super::types::ProcessedImage;

// ==========================================
// IMAGE PROCESSOR CONFIGURATION
// ==========================================

#[derive(Debug, Clone)]
pub struct ImageProcessorConfig {
    pub width: u32,
    pub height: u32,

    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            width: 128,
            height: 128,

            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

// ==========================================
// IMAGE PROCESSOR
// ==========================================

pub struct ImageProcessor {
    config: ImageProcessorConfig,
}

impl ImageProcessor {
    // ======================================
    // CREATE IMAGE PROCESSOR
    // ======================================

    pub fn new(config: ImageProcessorConfig) -> Result<Self, String> {
        if config.width == 0 || config.height == 0 {
            return Err("Image dimensions must be greater than zero".to_string());
        }

        for std in config.std {
            if !std.is_finite() || std <= 0.0 {
                return Err(
                    "Normalization standard deviation must be positive and finite".to_string(),
                );
            }
        }

        for value in config.mean {
            if !value.is_finite() {
                return Err("Normalization mean must be finite".to_string());
            }
        }

        Ok(Self { config })
    }

    // ======================================
    // PROCESS IMAGE FROM BYTES
    // ======================================

    pub fn process_bytes(&self, bytes: &[u8]) -> Result<ProcessedImage, String> {
        if bytes.is_empty() {
            return Err("Image data cannot be empty".to_string());
        }

        let image =
            image::load_from_memory(bytes).map_err(|e| format!("Failed to decode image: {}", e))?;

        // let hash = Sha256::digest(bytes);

        // println!("Image byte size: {}", bytes.len());
        // println!("Image SHA256: {:x}", hash);

        self.process_image(image)
    }

    // ======================================
    // PROCESS IMAGE FROM FILE
    // ======================================

    // pub fn process_file(&self, path: impl AsRef<Path>) -> Result<ProcessedImage, String> {
    //     let path = path.as_ref();

    //     let reader = ImageReader::open(path)
    //         .map_err(|e| format!("Failed to open image file '{}': {}", path.display(), e))?;

    //     let reader = reader
    //         .with_guessed_format()
    //         .map_err(|e| format!("Failed to determine image format: {}", e))?;

    //     let image = reader
    //         .decode()
    //         .map_err(|e| format!("Failed to decode image file '{}': {}", path.display(), e))?;

    //     self.process_image(image)
    // }

    // ======================================
    // MAIN PREPROCESSING PIPELINE
    // ======================================

    fn process_image(&self, image: DynamicImage) -> Result<ProcessedImage, String> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Validate tensor allocation size.

        let plane_size = width
            .checked_mul(height)
            .ok_or_else(|| "Image dimensions exceed supported size".to_string())?;

        let tensor_size = plane_size
            .checked_mul(3)
            .ok_or_else(|| "Image tensor dimensions exceed supported size".to_string())?;

        // Resize image.

        let resized =
            image.resize_exact(self.config.width, self.config.height, FilterType::Triangle);

        // Convert to RGB.

        let rgb = resized.to_rgb8();

        // Allocate NCHW tensor.

        let mut data = vec![0.0_f32; tensor_size];

        // Convert HWC RGB pixels to NCHW.

        for (x, y, pixel) in rgb.enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;

            let pixel_index = y * width + x;

            for channel in 0..3 {
                let pixel_value = pixel[channel] as f32 / 255.0;

                let normalized =
                    (pixel_value - self.config.mean[channel]) / self.config.std[channel];

                let tensor_index = channel * plane_size + pixel_index;

                data[tensor_index] = normalized;
            }
        }

        println!("Image processed: {}x{}", width, height);

        Ok(ProcessedImage {
            data,
            shape: [1, 3, height, width],
        })
    }

    // ======================================
    // ACCESS CONFIGURATION
    // ======================================

    // pub fn config(&self) -> &ImageProcessorConfig {
    //     &self.config
    // }
}
