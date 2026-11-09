
# Build stage
FROM rust:1-trixie AS builder

WORKDIR /app

# Install native build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

# Runtime stage
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/image-analysis-mcp-server \
    /usr/local/bin/image-analysis-mcp-server

RUN useradd --create-home appuser

USER appuser

ENTRYPOINT ["image-analysis-mcp-server"]
