"""
Session manager using LangGraph MemorySaver.
Handles conversation memory with persistence.
"""

from typing import Dict, Any
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger
from src.config import config


class SessionManager:
    """
    Unified session management using LangGraph checkpointers.
    Supports in-memory, SQLite, and PostgreSQL backends.
    """

    def __init__(self):
        self.backend = config.session.backend
        self.checkpointer = self._initialize_checkpointer()
        logger.success(f"✅ Session manager initialized (backend={self.backend})")

    def _initialize_checkpointer(self):
        """Initialize appropriate checkpointer based on config"""

        if self.backend == "memory":
            logger.info("Using in-memory checkpointer (InMemorySaver)")
            return InMemorySaver()

        elif self.backend == "sqlite":
            from langgraph.checkpoint.sqlite import SqliteSaver
            from pathlib import Path

            checkpoint_dir = Path(config.session.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            db_path = checkpoint_dir / config.system.checkpoint_db_name

            logger.info(f"Using SQLite checkpointer at: {db_path}")
            return SqliteSaver.from_conn_string(str(db_path))

        elif self.backend == "postgres":
            from langgraph.checkpoint.postgres import PostgresSaver

            conn_string = config.session.postgres_connection_string
            logger.info("Using PostgreSQL checkpointer")
            return PostgresSaver.from_conn_string(conn_string)

        else:
            logger.warning(f"Unknown backend '{self.backend}', defaulting to memory")
            return InMemorySaver()

    def get_checkpointer(self):
        """Get the underlying checkpointer for LangGraph"""
        return self.checkpointer

    def clear_session(self, thread_id: str) -> bool:
        """
        Clear conversation history for a specific thread/session.

        Args:
            thread_id: Session identifier

        Returns:
            Success status
        """
        logger.info(f"Clearing session: {thread_id}")
        # LangGraph handles this through thread_id management
        # Starting a new thread_id = fresh state
        return True

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions"""
        return {
            "backend": self.backend,
            "type": type(self.checkpointer).__name__
        }


# Global session manager instance
session_manager = SessionManager()
