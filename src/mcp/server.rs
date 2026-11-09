use axum::{Router, routing::get};

use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};

use std::sync::Arc;

use crate::tools::classification::backend::ClassificationBackend;

use super::handlers::ImageAnalysisServer;
use super::metadata::{health, root};

// ==========================================
// START MCP SERVER
// ==========================================

pub async fn run(
    classification_backend: Option<Arc<dyn ClassificationBackend>>,
    classification_labels: Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize MCP tool handler

    let mut server = ImageAnalysisServer::new();

    if let Some(backend) = classification_backend {
        server = server.with_classification_backend(backend, classification_labels);
    }

    // 2. Initialize Streamable HTTP service

    let mcp_service = StreamableHttpService::new(
        move || Ok(server.clone()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    // 3. Configure Axum routes

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .nest_service("/mcp", mcp_service);

    // 4. Configure server address

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());

    let address = format!("0.0.0.0:{}", port);

    // 5. Start HTTP server

    let listener = tokio::net::TcpListener::bind(&address).await?;

    println!("Image Analysis MCP Server running at {}", address);

    axum::serve(listener, app).await?;

    Ok(())
}
