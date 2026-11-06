
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

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use std::sync::Arc;

use crate::processors::classifier::ClassificationBackend;

use crate::tools::classification::{
    ClassificationParams,
    ClassificationTool,
};

// ==========================================
// MCP SERVER
// ==========================================

#[derive(Clone)]
pub struct ImageAnalysisServer {

    // Shared inference backend.
    // None until an inference engine is initialized.
    backend: Option<Arc<dyn ClassificationBackend>>,

    // Model class labels.
    labels: Vec<String>,
}

// ==========================================
// SERVER INITIALIZATION
// ==========================================

impl ImageAnalysisServer {

    pub fn new() -> Self {
        Self {
            backend: None,
            labels: Vec::new(),
        }
    }

    // Configure inference backend later.
    pub fn with_backend(
        mut self,
        backend: Arc<dyn ClassificationBackend>,
        labels: Vec<String>,
    ) -> Self {
        self.backend = Some(backend);
        self.labels = labels;

        self
    }
}

// ==========================================
// MCP TOOL REGISTRATION
// ==========================================

#[tool_router(server_handler)]
impl ImageAnalysisServer {

    #[tool(
        description = "Classify an image using the configured image classification model. Accepts a Base64-encoded JPEG, PNG or WebP image and returns the top three predictions with probabilities."
    )]
    fn classify_image(
        &self,
        Parameters(params): Parameters<ClassificationParams>,
    ) -> Result<String, String> {

        let response = ClassificationTool::classify(
            &params.image_base64,
            self.backend.as_ref(),
            &self.labels,
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
    })
}

// ==========================================
// SERVER RUNNER
// ==========================================

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {

    // Create MCP server
    let mcp_service = StreamableHttpService::new(
        || Ok(ImageAnalysisServer::new()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    // Configure Axum
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .nest_service("/mcp", mcp_service);

    // Configure port
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string());

    let address = format!("0.0.0.0:{}", port);

    // Start HTTP listener
    let listener = tokio::net::TcpListener::bind(&address).await?;

    println!("========================================");
    println!(" Image Analysis MCP Server");
    println!("========================================");
    println!(" Server:       http://localhost:{}", port);
    println!(" Health:       http://localhost:{}/health", port);
    println!(" MCP:          http://localhost:{}/mcp", port);
    println!(" MCP Tools:    classify_image");
    println!("========================================");

    axum::serve(listener, app).await?;

    Ok(())
}
