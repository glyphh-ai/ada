//! Glyphh runtime — single binary entry point.
//!
//! Wires together: runtime-api + ui-host, starts the Axum server.

use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .json()
        .init();

    tracing::info!("Starting Glyphh Runtime...");

    // TODO: Parse CLI args (port, db url, model dir, etc.)
    // TODO: Initialize database connection pool
    // TODO: Build Axum router from runtime-api + ui-host
    // TODO: Start server

    tracing::info!("Glyphh Runtime ready");
}
