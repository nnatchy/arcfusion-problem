#!/bin/bash
# Run Streamlit playground

echo "🎨 Starting ArcFusion RAG Streamlit Playground..."
echo "🌐 URL: http://localhost:8501"
echo ""

uv run streamlit run src/ui/streamlit_app.py --server.port 8501
