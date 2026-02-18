//! UI host: serves embedded Astro assets, SSE/WS event streaming.
//!
//! Replaces the separate studio container with an embedded static file server.

// TODO:
// - Embed Astro build output via include_dir or rust-embed
// - Serve static assets at /
// - SSE endpoint for real-time model events
// - WebSocket endpoint for interactive sessions
