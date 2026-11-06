# Build stage
FROM rust:1-bookworm AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install CA certificates for secure outbound API/HTTPS calls
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled binary
COPY --from=builder /app/target/release/image-analysis-mcp-server /usr/local/bin/image-analysis-mcp-server

# Run as non-root user
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["image-analysis-mcp-server"]