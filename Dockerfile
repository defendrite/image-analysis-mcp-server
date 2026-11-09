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

# 2. Force-create ALL possible variants of the cache directories and make them universally writable
RUN mkdir -p /home/defendrite/.cache/huggingface \
             /home/defendrite/.cache/huggingface/hub \
             /app/.cache \
             /tmp/huggingface && \
    chmod -R 777 /home/defendrite/.cache /app/.cache /tmp/huggingface && \
    chown -R defendrite:defendrite /home/defendrite/.cache /app/.cache /tmp/huggingface

# 3. Explicitly flood the environment with every fallback variable the Rust ecosystem checks
ENV HF_HOME=/home/defendrite/.cache/huggingface
ENV HF_HUB_CACHE=/home/defendrite/.cache/huggingface/hub
ENV XDG_CACHE_HOME=/home/defendrite/.cache

USER defendrite

ENTRYPOINT ["image-analysis-mcp-server"]
