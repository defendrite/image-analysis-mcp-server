mod server;
mod models;
mod processors;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    server::run().await?;

    Ok(())
}
