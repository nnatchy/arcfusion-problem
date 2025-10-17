"""Pydantic models for FastAPI request/response schemas"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class QueryStatus(str, Enum):
    """Status of query processing"""
    success = "success"
    blocked = "blocked"
    clarification_needed = "clarification_needed"
    error = "error"


class Source(BaseModel):
    """Source information for citations"""
    type: str = Field(..., description="Source type: 'pdf' or 'web'")

    # PDF fields
    source: Optional[str] = Field(None, description="PDF filename or paper title")
    page: Optional[str] = Field(None, description="Page number")

    # Web fields
    title: Optional[str] = Field(None, description="Web page title")
    url: Optional[str] = Field(None, description="Web page URL")
    date: Optional[str] = Field(None, description="Publication date")

    # Common fields
    score: float = Field(..., description="Relevance score")
    text_preview: Optional[str] = Field(None, description="Preview of source text")


class AskRequest(BaseModel):
    """Request model for /ask endpoint"""
    query: str = Field(..., description="User query", min_length=1, max_length=2000)
    session_id: Optional[str] = Field("default", description="Session ID for conversation history")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What were the main findings in the Zhang et al. 2024 paper about text-to-SQL?",
                "session_id": "user_123"
            }
        }


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""
    status: QueryStatus = Field(..., description="Query processing status")
    answer: Optional[str] = Field(None, description="Generated answer")
    sources: List[Source] = Field(default_factory=list, description="List of sources used")

    # Metadata
    quality_score: Optional[float] = Field(None, description="Quality score (0.0-1.0)")
    routing_decision: Optional[str] = Field(None, description="Routing decision: pdf_only/web_only/hybrid")
    iterations: Optional[int] = Field(None, description="Number of reflection iterations")

    # For blocked queries
    reason: Optional[str] = Field(None, description="Reason for blocking (if blocked)")

    # For clarification needed
    questions: Optional[List[str]] = Field(None, description="Clarification questions")
    issues: Optional[List[str]] = Field(None, description="Clarity issues found")

    # For errors
    error: Optional[str] = Field(None, description="Error message (if error)")

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "answer": "According to Zhang et al. 2024, the main findings...",
                "sources": [
                    {
                        "type": "pdf",
                        "source": "Zhang et al. - 2024 - Benchmarking the Text-to-SQL Capability.pdf",
                        "page": "5",
                        "score": 0.92,
                        "text_preview": "We evaluated GPT-4 on multiple benchmarks..."
                    }
                ],
                "quality_score": 0.95,
                "routing_decision": "pdf_only",
                "iterations": 1,
                "metadata": {
                    "is_complex": False,
                    "routing_confidence": 0.98,
                    "num_sources": 3
                }
            }
        }


class ClearMemoryRequest(BaseModel):
    """Request model for /clear_memory endpoint"""
    session_id: str = Field(..., description="Session ID to clear")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user_123"
            }
        }


class ClearMemoryResponse(BaseModel):
    """Response model for /clear_memory endpoint"""
    success: bool = Field(..., description="Whether memory was cleared successfully")
    message: str = Field(..., description="Status message")
    session_id: str = Field(..., description="Session ID that was cleared")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Memory cleared for session: user_123",
                "session_id": "user_123"
            }
        }


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Status of each component")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "components": {
                    "llm": "operational",
                    "embeddings": "operational",
                    "vector_store": "operational",
                    "orchestrator": "operational"
                }
            }
        }
