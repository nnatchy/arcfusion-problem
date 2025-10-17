# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (curl for uv, build tools for wheels, healthchecks)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv directly to /usr/local/bin (system-wide)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv && \
    cp /root/.local/bin/uvx /usr/local/bin/uvx && \
    uv --version

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock .python-version ./

# Sync dependencies using uv (before copying source code for better caching)
RUN uv sync --frozen

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config.ini ./

# Create logs directory
RUN mkdir -p logs

# Expose API port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
