"""FastAPI application entry point for ArcFusion RAG System"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import config
from src.utils.logging_config import setup_logging
from src.api.routes import router

# Setup logging
setup_logging(config)

# Create FastAPI app
app = FastAPI(
    title="ArcFusion RAG System",
    description="Multi-agent RAG system for academic papers with recursive retrieval and web search",
    version=config.system.version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 80)
    logger.info("🚀 Starting ArcFusion RAG System")
    logger.info("=" * 80)
    logger.info(f"LLM Model: {config.llm.model}")
    logger.info(f"Embedding Model: {config.embeddings.model}")
    logger.info(f"Vector Store: {config.pinecone.index_name}")
    logger.info(f"Max Recursion Depth: {config.rag.max_recursion_depth}")
    logger.info(f"Reflection Enabled: {config.agents.enable_reflection}")
    logger.info(f"Security Check Enabled: {config.agents.enable_security_check}")
    logger.info("=" * 80)
    logger.success("✅ ArcFusion RAG System ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down ArcFusion RAG System - cleaning up resources")

    # Clear reranker model to prevent semaphore leak
    try:
        from src.retrieval.vector_store import get_vector_store
        vector_store = get_vector_store()
        if hasattr(vector_store, 'reranker') and vector_store.reranker:
            logger.info("Unloading reranker model...")
            del vector_store.reranker.reranker  # FlagReranker instance
            del vector_store.reranker
            logger.info("✅ Reranker model unloaded")
    except Exception as e:
        logger.warning(f"Could not cleanup reranker: {e}")

    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✅ CUDA cache cleared")
    except ImportError:
        pass  # torch not available, skip
    except Exception as e:
        logger.warning(f"Could not clear CUDA cache: {e}")

    logger.info("✅ Shutdown cleanup complete")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ArcFusion RAG System API",
        "version": config.system.version,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
        log_level="info"
    )
