#!/bin/bash
# Run FastAPI server

# Read config values using Python
API_HOST=$(python -c "from src.config import config; print(config.api.host)" 2>/dev/null || echo "0.0.0.0")
API_PORT=$(python -c "from src.config import config; print(config.api.port)" 2>/dev/null || echo "8000")

echo "🚀 Starting ArcFusion RAG API Server..."
echo "📚 API Docs: http://localhost:${API_PORT}/docs"
echo "🔍 ReDoc: http://localhost:${API_PORT}/redoc"
echo ""

uv run uvicorn src.main:app --reload --host ${API_HOST} --port ${API_PORT}
