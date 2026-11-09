use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct HealthResponse {
    status: &'static str,
    service: &'static str,
}

#[derive(Serialize)]
pub struct ServerInfo {
    name: &'static str,
    version: &'static str,
    description: &'static str,
    transport: &'static str,
    endpoints: Endpoints,
    tools: Vec<&'static str>,
}

#[derive(Serialize)]
pub struct Endpoints {
    health: &'static str,
    mcp: &'static str,
}

// ==========================================
// HEALTH ENDPOINT
// ==========================================

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "image-analysis-mcp-server",
    })
}

// ==========================================
// SERVER INFORMATION ENDPOINT
// ==========================================

pub async fn root() -> Json<ServerInfo> {
    Json(ServerInfo {
        name: "image-analysis-mcp-server",
        version: env!("CARGO_PKG_VERSION"),
        description: "Modular Image Analysis MCP Server",
        transport: "streamable-http",

        endpoints: Endpoints {
            health: "/health",
            mcp: "/mcp",
        },

        tools: vec!["classify_dog_or_cat"],
    })
}
