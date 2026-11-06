# image-analysis-mcp-server

Image Analysis MCP Server

Build and Run

```bash
cargo build --release
./target/release/image-analysis-mcp-server
```

Pull Docker Image and Run

```bash
docker pull ghcr.io/defendrite/image-analysis-mcp-server:latest
docker run -e PORT=9002 -p 9002:9002 ghcr.io/defendrite/image-analysis-mcp-server:latest
```

Tools :
`say-hello`

## Technologies

- Rust
- rmcp
- Tokio
- Serde
- Model Context Protocol (MCP)

## Resources

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Rust](https://www.rust-lang.org/)
- [rmcp SDK](https://github.com/modelcontextprotocol/rust-sdk)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
