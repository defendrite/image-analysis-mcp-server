# Runtime stage
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/image-analysis-mcp-server \
    /usr/local/bin/image-analysis-mcp-server

# 1. Create the system defendrite
RUN useradd --create-home defendrite

# 2. MATCH YOUR RUST CODE: Pre-create /app/models_cache and give defendrite full ownership
RUN mkdir -p /app/models_cache && \
    chown -R defendrite:defendrite /app/models_cache && \
    chmod -R 755 /app/models_cache

USER defendrite

ENTRYPOINT ["image-analysis-mcp-server"]
