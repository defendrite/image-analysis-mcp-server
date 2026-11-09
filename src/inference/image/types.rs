#[derive(Debug, Clone)]
pub struct ProcessedImage {
    // NCHW format: [1, 3, height, width]
    pub data: Vec<f32>,

    pub shape: [usize; 4],
}
