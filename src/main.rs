mod app;
mod inference;
mod mcp;
mod models;
mod tools;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::bootstrap::run().await
}
