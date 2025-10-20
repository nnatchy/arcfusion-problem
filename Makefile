.PHONY: help build up down restart logs logs-api logs-streamlit shell shell-api test clean install dev-setup check ingest

# Default target - show help
help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║          ArcFusion RAG System - Makefile Commands              ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 Docker Commands:"
	@echo "  make build          - Build Docker images"
	@echo "  make up             - Start all services (API + Streamlit)"
	@echo "  make down           - Stop and remove all containers"
	@echo "  make restart        - Restart all services"
	@echo "  make logs           - View logs from all services"
	@echo "  make logs-api       - View API logs only"
	@echo "  make logs-streamlit - View Streamlit logs only"
	@echo ""
	@echo "🔧 Development Commands:"
	@echo "  make install        - Install dependencies locally with uv"
	@echo "  make dev-setup      - Setup development environment (.env + install)"
	@echo "  make shell          - Enter API container shell"
	@echo "  make shell-api      - Enter API container shell (alias)"
	@echo ""
	@echo "🧪 Testing Commands:"
	@echo "  make test              - Run component tests"
	@echo "  make test-components   - Test core components"
	@echo "  make test-simple       - Run simple end-to-end test"
	@echo "  make test-eval         - Run Golden Q&A evaluation (Bonus #2)"
	@echo "  make test-eval-threshold - Check if evaluation meets 70%% threshold"
	@echo "  make test-all          - Run all tests (components + simple + eval)"
	@echo "  make test-memory       - Test history compression"
	@echo "  make test-perf         - Run performance benchmarks"
	@echo "  make check             - Check system status"
	@echo ""
	@echo "📚 Data Commands:"
	@echo "  make ingest         - Ingest PDFs to Pinecone (local)"
	@echo "  make ingest-docker  - Ingest PDFs using Docker"
	@echo ""
	@echo "🧹 Cleanup Commands:"
	@echo "  make clean          - Remove logs, checkpoints, cache"
	@echo "  make clean-docker   - Remove Docker images and volumes"
	@echo "  make clean-all      - Clean everything (local + Docker)"
	@echo ""

# ============================================
# Docker Commands
# ============================================

build:
	@echo "🔨 Building Docker images..."
	docker compose build

up:
	@echo "🚀 Starting all services..."
	docker compose up
	@echo "✅ Services started!"
	@echo "   API: http://localhost:8000"
	@echo "   API Docs: http://localhost:8000/docs"
	@echo "   Streamlit: http://localhost:8501"
	@echo ""
	@echo "💡 View logs: make logs"

down:
	@echo "🛑 Stopping all services..."
	docker compose down
	@echo "✅ Services stopped!"

restart:
	@echo "🔄 Restarting all services..."
	docker compose restart
	@echo "✅ Services restarted!"

logs:
	@echo "📋 Following logs from all services (Ctrl+C to exit)..."
	docker compose logs -f

logs-api:
	@echo "📋 Following API logs (Ctrl+C to exit)..."
	docker compose logs -f api

logs-streamlit:
	@echo "📋 Following Streamlit logs (Ctrl+C to exit)..."
	docker compose logs -f streamlit

shell:
	@echo "🐚 Entering API container shell..."
	docker compose exec api /bin/bash

shell-api:
	@echo "🐚 Entering API container shell..."
	docker compose exec api /bin/bash

# ============================================
# Development Commands
# ============================================

install:
	@echo "📦 Installing dependencies with uv..."
	uv sync
	@echo "✅ Dependencies installed!"

dev-setup:
	@echo "🔧 Setting up development environment..."
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "⚠️  Please edit .env with your API keys!"; \
	else \
		echo "✅ .env file already exists"; \
	fi
	@echo ""
	@echo "📦 Installing dependencies..."
	uv sync
	@echo ""
	@echo "✅ Development environment ready!"
	@echo "💡 Next steps:"
	@echo "   1. Edit .env with your API keys"
	@echo "   2. Run 'make test-components' to verify setup"
	@echo "   3. Run 'make ingest' to load PDFs to Pinecone"

check:
	@echo "🔍 Checking system status..."
	@echo ""
	@echo "📦 Docker:"
	@docker --version 2>/dev/null || echo "  ❌ Docker not installed"
	@docker compose version 2>/dev/null && echo "  ✅ Docker Compose available" || docker-compose --version 2>/dev/null && echo "  ✅ Docker Compose available" || echo "  ❌ Docker Compose not installed"
	@docker ps >/dev/null 2>&1 && echo "  ✅ Docker daemon running" || echo "  ❌ Docker daemon not running (run: open -a Docker)"
	@echo ""
	@echo "🐍 Python Environment:"
	@uv --version 2>/dev/null && echo "  ✅ uv installed" || echo "  ❌ uv not installed"
	@python3 --version 2>/dev/null || echo "  ❌ Python not found"
	@echo ""
	@echo "📄 Configuration Files:"
	@[ -f .env ] && echo "  ✅ .env exists" || echo "  ⚠️  .env missing (run 'make dev-setup')"
	@[ -f config.ini ] && echo "  ✅ config.ini exists" || echo "  ❌ config.ini missing"
	@[ -f pyproject.toml ] && echo "  ✅ pyproject.toml exists" || echo "  ❌ pyproject.toml missing"
	@[ -f uv.lock ] && echo "  ✅ uv.lock exists" || echo "  ❌ uv.lock missing"
	@echo ""
	@echo "🐳 Docker Services:"
	@docker compose ps 2>/dev/null || echo "  No services running"

# ============================================
# Testing Commands
# ============================================

test: test-components

test-components:
	@echo "🧪 Running component tests..."
	uv run python scripts/test_components.py

test-simple:
	@echo "🧪 Running simple end-to-end test..."
	uv run python scripts/simple_test.py

test-memory:
	@echo "🧪 Testing history compression..."
	uv run python scripts/test_history_compression.py

test-perf:
	@echo "🧪 Running performance benchmarks..."
	uv run python scripts/performance_test.py

test-eval:
	@echo "🧪 Running Golden Q&A evaluation (Bonus #2)..."
	uv run python scripts/evaluate_golden_qa.py

test-eval-threshold:
	@echo "🎯 Checking evaluation threshold..."
	uv run python scripts/check_accuracy_threshold.py

test-all: test-components test-simple test-eval
	@echo "✅ All tests complete!"

# ============================================
# Data Commands
# ============================================

ingest:
	@echo "📚 Ingesting PDFs to Pinecone..."
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found. Run 'make dev-setup' first."; \
		exit 1; \
	fi
	uv run python scripts/ingest_pdfs.py

ingest-docker:
	@echo "📚 Ingesting PDFs using Docker..."
	docker compose exec api uv run python scripts/ingest_pdfs.py

# ============================================
# Cleanup Commands
# ============================================

clean:
	@echo "🧹 Cleaning local files..."
	rm -rf logs/*.log
	rm -rf checkpoints/*
	rm -rf .uv/
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✅ Local files cleaned!"

clean-docker:
	@echo "🧹 Cleaning Docker resources..."
	docker compose down -v --rmi all
	@echo "✅ Docker resources cleaned!"

clean-all: clean clean-docker
	@echo "✅ Everything cleaned!"

# ============================================
# Quick Workflows
# ============================================

# Full setup from scratch
setup: dev-setup
	@echo ""
	@echo "✅ Setup complete!"
	@echo "💡 Next: Edit .env, then run 'make ingest'"

# Build and start in one command
start: build up

# Complete restart (rebuild + up)
rebuild: down build up

# Quick test after changes
quick-test: test-components test-simple
