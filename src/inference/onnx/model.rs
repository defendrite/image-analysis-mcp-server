use ort::{environment::Environment, session::Session};

use std::sync::{Arc, Mutex};

pub struct OnnxModel {
    pub(crate) _environment: Arc<Environment>,
    pub(crate) session: Mutex<Session>,

    pub(crate) input_name: String,
    pub(crate) output_name: String,
    // pub(crate) device: ExecutionDevice,
}

impl OnnxModel {
    // pub fn device(&self) -> ExecutionDevice {
    //     self.device
    // }

    // pub fn selected_output_name(&self) -> &str {
    //     &self.output_name
    // }

    // pub fn input_names(&self) -> Result<Vec<String>, String> {
    //     let session = self
    //         .session
    //         .lock()
    //         .map_err(|_| "ONNX session mutex poisoned".to_string())?;

    //     Ok(session
    //         .inputs()
    //         .iter()
    //         .map(|input| input.name().to_string())
    //         .collect())
    // }

    // pub fn output_names(&self) -> Result<Vec<String>, String> {
    //     let session = self
    //         .session
    //         .lock()
    //         .map_err(|_| "ONNX session mutex poisoned".to_string())?;

    //     Ok(session
    //         .outputs()
    //         .iter()
    //         .map(|output| output.name().to_string())
    //         .collect())
    // }
}
