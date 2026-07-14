# Build stage — compile C++ project
FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY CMakeLists.txt .
COPY src/ src/
COPY tests/ tests/

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc) --config Release

# Runtime stage — lightweight
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages \
       fastapi uvicorn websockets python-dotenv httpx jinja2 aiofiles

WORKDIR /app

# Copy built binaries
COPY --from=builder /build/build/Release/train_gpt /app/build/Release/train_gpt
COPY --from=builder /build/build/Release/*.lib /app/build/Release/ 2>/dev/null || true

# Copy project files
COPY CMakeLists.txt .
COPY src/ src/
COPY web/ web/
COPY data/ data/
COPY docs/ docs/
COPY tests/ tests/
COPY shaders/ shaders/
COPY scripts/ scripts/

# Pre-build to warm up
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc) --config Release --target train_gpt

EXPOSE 8080

CMD ["python3", "web/server.py"]
