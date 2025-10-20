#!/bin/bash
# Run Streamlit playground

# Read config values using Python
STREAMLIT_PORT=$(python -c "from src.config import config; print(config.streamlit.port)" 2>/dev/null || echo "8501")

echo "🎨 Starting ArcFusion RAG Streamlit Playground..."
echo "🌐 URL: http://localhost:${STREAMLIT_PORT}"
echo ""

uv run streamlit run src/ui/streamlit_app.py --server.port ${STREAMLIT_PORT}
