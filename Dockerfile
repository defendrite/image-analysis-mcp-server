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

# 1. Create the user
RUN useradd --create-home defendrite

# 2. ADD THIS: Pre-create the Hugging Face cache directory and give defendrite ownership
RUN mkdir -p /home/defendrite/.cache/huggingface && chown -R defendrite:defendrite /home/defendrite/.cache

# 3. ADD THIS: Set the environment variable so Rust frameworks know where to write
ENV HF_HOME=/home/defendrite/.cache/huggingface

USER defendrite

ENTRYPOINT ["image-analysis-mcp-server"]
