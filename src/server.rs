
use axum::{
    Json,
    Router,
    routing::get,
};

use rmcp::{
    handler::server::wrapper::Parameters,
    tool,
    tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig,
        StreamableHttpService,
        session::local::LocalSessionManager,
    },
};

use serde::Serialize;
use std::sync::Arc;

use crate::processors::classifier::ClassificationBackend;

use crate::tools::classification::{
    ClassificationParams,
    ClassificationTool,
};

use crate::tools::age_detection::{
    AgeDetectionBackend,
    AgeDetectionTool,
};

// ==========================================
// MCP SERVER
// ==========================================

#[derive(Clone)]
pub struct ImageAnalysisServer {

    // Classification model
    classification_backend: Option<Arc<dyn ClassificationBackend>>,

    // Age detection model
    age_detection_backend: Option<Arc<dyn AgeDetectionBackend>>,

    // Classification labels
    classification_labels: Vec<String>,
}

// ==========================================
// SERVER INITIALIZATION
// ==========================================

impl ImageAnalysisServer {

    pub fn new() -> Self {
        Self {
            classification_backend: None,
            age_detection_backend: None,
            classification_labels: Vec::new(),
        }
    }

    // Configure classification model
    pub fn with_classification_backend(
        mut self,
        backend: Arc<dyn ClassificationBackend>,
        labels: Vec<String>,
    ) -> Self {
        self.classification_backend = Some(backend);
        self.classification_labels = labels;

        self
    }

    // Configure age detection model
    pub fn with_age_detection_backend(
        mut self,
        backend: Arc<dyn AgeDetectionBackend>,
    ) -> Self {
        self.age_detection_backend = Some(backend);

        self
    }
}

// ==========================================
// MCP TOOL REGISTRATION
// ==========================================

#[tool_router(server_handler)]
impl ImageAnalysisServer {

    // ======================================
    // IMAGE CLASSIFICATION
    // ======================================

    #[tool(
        description = "Classify an image using the configured classification model. Accepts a Base64-encoded JPEG, PNG or WebP image and returns the top three predictions."
    )]
    fn classify_image(
        &self,
        Parameters(params): Parameters<ClassificationParams>,
    ) -> Result<String, String> {

        let response = ClassificationTool::classify(
            &params.image_base64,
            self.classification_backend.as_ref(),
            &self.classification_labels,
        )?;

        serde_json::to_string_pretty(&response)
            .map_err(|e| e.to_string())
    }

    // ======================================
    // AGE DETECTION
    // ======================================

    #[tool(
        description = "Estimate the apparent age in years from a Base64-encoded JPEG, PNG or WebP image containing a visible face. Returns estimated age and optional model confidence."
    )]
    fn detect_age(
        &self,
        Parameters(params): Parameters<ClassificationParams>,
    ) -> Result<String, String> {

        let response = AgeDetectionTool::detect(
            &params.image_base64,
            self.age_detection_backend.as_ref(),
        )?;

        serde_json::to_string_pretty(&response)
            .map_err(|e| e.to_string())
    }
}

// ==========================================
// HTTP RESPONSE STRUCTURES
// ==========================================

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
}

#[derive(Serialize)]
struct ServerInfo {
    name: &'static str,
    version: &'static str,
    description: &'static str,
    transport: &'static str,
    endpoints: Endpoints,
    tools: Vec<&'static str>,
}

#[derive(Serialize)]
struct Endpoints {
    health: &'static str,
    mcp: &'static str,
}

// ==========================================
// HTTP ENDPOINTS
// ==========================================

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "image-analysis-mcp-server",
    })
}

async fn root() -> Json<ServerInfo> {
    Json(ServerInfo {
        name: "image-analysis-mcp-server",
        version: env!("CARGO_PKG_VERSION"),
        description: "Modular Image Analysis MCP Server",
        transport: "streamable-http",

        endpoints: Endpoints {
            health: "/health",
            mcp: "/mcp",
        },

        tools: vec![
            "classify_image",
            "detect_age",
        ],
    })
}

// ==========================================
// SERVER RUNNER
// ==========================================

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {

    let mcp_service = StreamableHttpService::new(
        || Ok(ImageAnalysisServer::new()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .nest_service("/mcp", mcp_service);

    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string());

    let address = format!("0.0.0.0:{}", port);

    let listener = tokio::net::TcpListener::bind(&address).await?;

    println!("========================================");
    println!(" Image Analysis MCP Server");
    println!("========================================");
    println!(" Server:       http://localhost:{}", port);
    println!(" Health:       http://localhost:{}/health", port);
    println!(" MCP:          http://localhost:{}/mcp", port);
    println!(" MCP Tools:");
    println!("  1. classify_image");
    println!("  2. detect_age");
    println!("========================================");

    axum::serve(listener, app).await?;

    Ok(())
}
