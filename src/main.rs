use axum::{Json, Router, routing::get};
use rmcp::{
    handler::server::wrapper::Parameters,
    schemars, tool, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde::{Deserialize, Serialize};

// -------------------------
// Tool Parameters
// -------------------------
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct HelloParams {
    name: String,
}
// -------------------------
// MCP Server
// -------------------------

#[derive(Clone)]
struct ImageAnalysisServer;

#[tool_router(server_handler)]
impl ImageAnalysisServer {
    #[tool(description = "Say hello to a user")]
    fn say_hello(
        &self,
        Parameters(HelloParams { name }): Parameters<HelloParams>,
    ) -> Result<String,String> {
        Ok(format!("Hello, {}! Welcome to Rust MCP Server.", name))
    }
}

// -------------------------
// HTTP Response Models
// -------------------------

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

// -------------------------
// HTTP Endpoints
// -------------------------

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "image-analysis-mcp-server",
    })
}

async fn root() -> Json<ServerInfo> {
    Json(ServerInfo {
        name: "image-analysis-mcp-server",
        version: "1.0.0",
        description: "Image Analysis MCP Server ",
        transport: "streamable-http",
        endpoints: Endpoints {
            health: "/health",
            mcp: "/mcp",
        },
    })
}

// -------------------------
// Main
// -------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure MCP HTTP service
    let mcp_service = StreamableHttpService::new(
        || Ok(ImageAnalysisServer),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    // Build Axum application
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .nest_service("/mcp", mcp_service);

    // Start HTTP server
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let address = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&address).await?;

    eprintln!("MCP Server started successfully!");
    eprintln!("HTTP server listening on {}", &address);
    eprintln!("Health endpoint: http://localhost:{}/health",port);
    eprintln!("MCP endpoint: http://localhost:{}/mcp",port);

    axum::serve(listener, app).await?;

    Ok(())
}
