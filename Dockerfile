# Build stage
FROM rust:1-trixie AS builder
WORKDIR /app
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
COPY --from=builder /app/target/release/image-analysis-mcp-server /usr/local/bin/image-analysis-mcp-server

RUN useradd --create-home defendrite

# Set a safe, universally writable default path inside the container
ENV MODEL_CACHE_DIR=/tmp/models_cache

USER defendrite
ENTRYPOINT ["image-analysis-mcp-server"]
