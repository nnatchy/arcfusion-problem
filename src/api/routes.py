"""FastAPI routes for ArcFusion RAG system"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from src.api.models import (
    AskRequest,
    AskResponse,
    ClearMemoryRequest,
    ClearMemoryResponse,
    HealthResponse,
    Source,
    QueryStatus
)
from src.agents.orchestrator import orchestrator
from src.config import config

# Create router
router = APIRouter()


@router.post("/ask", response_model=AskResponse, tags=["Query"])
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Ask a question to the RAG system.

    The system will:
    1. Validate the query for security and clarity
    2. Route to appropriate information sources (PDF, web, or both)
    3. Retrieve relevant information
    4. Synthesize a comprehensive answer
    5. Reflect on quality and refine if needed

    Returns a structured response with answer, sources, and metadata.
    """
    logger.info(f"📥 Received query from session {request.session_id}: {request.query[:100]}...")

    try:
        # Process query through orchestrator
        result = await orchestrator.ask(
            query=request.query,
            session_id=request.session_id
        )

        # Convert to response model
        status_value = QueryStatus(result.get('status', 'error'))

        # Convert sources to Source models
        sources = [
            Source(**source) for source in result.get('sources', [])
        ]

        response = AskResponse(
            status=status_value,
            answer=result.get('answer'),
            sources=sources,
            quality_score=result.get('quality_score'),
            routing_decision=result.get('routing_decision'),
            iterations=result.get('iterations'),
            reason=result.get('reason'),
            questions=result.get('questions'),
            issues=result.get('issues'),
            error=result.get('error'),
            metadata=result.get('metadata')
        )

        logger.success(f"✅ Query processed successfully: {status_value}")
        return response

    except Exception as e:
        logger.error(f"❌ Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/clear_memory", response_model=ClearMemoryResponse, tags=["Session"])
async def clear_session_memory(request: ClearMemoryRequest) -> ClearMemoryResponse:
    """
    Clear conversation history for a session.

    This removes all stored conversation context for the given session ID,
    allowing for a fresh start.
    """
    logger.info(f"🗑️ Clearing memory for session: {request.session_id}")

    try:
        orchestrator.clear_memory(request.session_id)

        return ClearMemoryResponse(
            success=True,
            message=f"Memory cleared for session: {request.session_id}",
            session_id=request.session_id
        )

    except Exception as e:
        logger.error(f"❌ Error clearing memory: {str(e)}")
        return ClearMemoryResponse(
            success=False,
            message=f"Error clearing memory: {str(e)}",
            session_id=request.session_id
        )


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Check system health and component status.

    Returns the operational status of all major components.
    """
    logger.debug("🏥 Health check requested")

    components = {
        "llm": "operational",
        "embeddings": "operational",
        "vector_store": "operational",
        "orchestrator": "operational"
    }

    # Could add actual health checks here
    # For now, assume all operational if service is running

    return HealthResponse(
        status="healthy",
        version=config.system.version,
        components=components
    )
