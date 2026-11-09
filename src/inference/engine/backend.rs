#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, String> {
        if shape.is_empty() {
            return Err("Tensor shape cannot be empty".to_string());
        }

        if shape.iter().any(|&dim| dim == 0) {
            return Err("Tensor dimensions must be greater than zero".to_string());
        }

        let expected_size = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "Tensor dimensions overflow".to_string())?;

        if data.len() != expected_size {
            return Err(format!(
                "Tensor size mismatch: expected {}, received {}",
                expected_size,
                data.len()
            ));
        }

        if data.iter().any(|value| !value.is_finite()) {
            return Err("Tensor contains NaN or infinite values".to_string());
        }

        Ok(Self { shape, data })
    }
}

pub trait InferenceBackend: Send + Sync {
    fn infer(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String>;
}
