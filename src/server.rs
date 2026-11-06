
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

// ==========================================
// MCP SERVER
// ==========================================

#[derive(Clone)]
pub struct ImageAnalysisServer;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct HelloParams {
    pub name: String,
}

#[tool_router(server_handler)]
impl ImageAnalysisServer {

    #[tool(description = "Say hello to a user")]
    fn say_hello(
        &self,
        Parameters(HelloParams { name }): Parameters<HelloParams>,
    ) -> String {
        format!(
            "Hello, {}! Welcome to Image Analysis MCP Server.",
            name
        )
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
// SERVER INITIALIZATION
// ==========================================

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {

    // Configure MCP service
    let mcp_service = StreamableHttpService::new(
        || Ok(ImageAnalysisServer),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    // Configure Axum routes
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .nest_service("/mcp", mcp_service);

    // Read port from environment
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string());

    let address = format!("0.0.0.0:{}", port);

    // Bind TCP listener
    let listener = tokio::net::TcpListener::bind(&address).await?;

    println!("========================================");
    println!(" Image Analysis MCP Server");
    println!("========================================");
    println!(" Server:  http://localhost:{}", port);
    println!(" Health:  http://localhost:{}/health", port);
    println!(" MCP:     http://localhost:{}/mcp", port);
    println!("========================================");

    // Start server
    axum::serve(listener, app).await?;

    Ok(())
}
