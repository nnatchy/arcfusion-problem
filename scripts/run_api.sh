#!/bin/bash
# Run FastAPI server

echo "🚀 Starting ArcFusion RAG API Server..."
echo "📚 API Docs: http://localhost:8000/docs"
echo "🔍 ReDoc: http://localhost:8000/redoc"
echo ""

uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
