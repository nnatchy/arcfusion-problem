"""
History Manager - Token-based conversation history compression with intelligent summarization.

Handles:
- Token counting using tiktoken
- Automatic summarization when token threshold is exceeded
- Academic-focused context preservation (papers, authors, findings)
- Sliding window for recent messages
"""

import json
from typing import List, Dict, Any, Optional
from loguru import logger
from src.core.llm_client import llm_client
from src.config import config

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using character-based estimation")

class HistoryManager:
    """Manages conversation history with token-based compression"""

    def __init__(self):
        # Initialize tiktoken encoder for accurate token counting
        if TIKTOKEN_AVAILABLE:
            try:
                # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo)
                self.encoder = tiktoken.get_encoding("cl100k_base")
                logger.info("✅ Tiktoken encoder initialized for accurate token counting")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}, using estimation")
                self.encoder = None
        else:
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        Falls back to character-based estimation if tiktoken unavailable.
        """
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                logger.warning(f"Token counting error: {e}, using estimation")

        # Fallback: estimate 4 characters per token (rough approximation)
        return len(text) // 4

    def count_history_tokens(self, history: List[Dict[str, Any]]) -> int:
        """Count total tokens in conversation history"""
        total = 0
        for msg in history:
            content = str(msg.get('content', ''))
            total += self.count_tokens(content)
        return total

    async def compress_history(
        self,
        history: List[Dict[str, Any]],
        max_tokens: int = None,
        keep_recent_tokens: int = None
    ) -> Dict[str, Any]:
        """
        Compress conversation history using token-based windowing and summarization.

        Args:
            history: Full conversation history
            max_tokens: Maximum tokens before triggering summarization (from config if None)
            keep_recent_tokens: Tokens to keep in recent window (from config if None)

        Returns:
            Dict with:
                - summary: str (compressed summary of old messages)
                - recent: List[Dict] (recent messages)
                - total_tokens: int (token count after compression)
                - compressed: bool (whether compression was applied)
        """
        if not history:
            return {
                "summary": "",
                "recent": [],
                "total_tokens": 0,
                "compressed": False
            }

        # Get config values
        if max_tokens is None:
            max_tokens = getattr(config.memory, 'max_history_tokens', 10000)
        if keep_recent_tokens is None:
            keep_recent_tokens = getattr(config.memory, 'keep_recent_tokens', 5000)

        # Count total tokens
        total_tokens = self.count_history_tokens(history)

        logger.info(f"📊 History stats: {len(history)} messages, {total_tokens} tokens")

        # No compression needed
        if total_tokens <= max_tokens:
            logger.info(f"✅ History within limit ({total_tokens}/{max_tokens} tokens), no compression needed")
            return {
                "summary": "",
                "recent": history,
                "total_tokens": total_tokens,
                "compressed": False
            }

        # Need to compress - find split point
        logger.info(f"⚠️ History exceeds limit ({total_tokens}/{max_tokens} tokens), compressing...")

        # Keep recent messages that fit in keep_recent_tokens
        recent_messages = []
        recent_tokens = 0

        # Work backwards from most recent
        for msg in reversed(history):
            msg_tokens = self.count_tokens(str(msg.get('content', '')))
            if recent_tokens + msg_tokens <= keep_recent_tokens:
                recent_messages.insert(0, msg)  # Insert at beginning to maintain order
                recent_tokens += msg_tokens
            else:
                break

        # Messages to summarize
        old_messages = history[:len(history) - len(recent_messages)]

        if not old_messages:
            # Edge case: even the most recent message exceeds keep_recent_tokens
            logger.warning("Single message exceeds token limit, keeping it anyway")
            return {
                "summary": "",
                "recent": history[-1:],  # Keep just the last message
                "total_tokens": self.count_history_tokens(history[-1:]),
                "compressed": False
            }

        logger.info(f"📝 Summarizing {len(old_messages)} old messages, keeping {len(recent_messages)} recent")

        # Generate summary
        summary = await self._generate_summary(old_messages)

        summary_tokens = self.count_tokens(summary)
        final_total = summary_tokens + recent_tokens

        logger.success(
            f"✅ Compression complete: {total_tokens} → {final_total} tokens "
            f"(summary: {summary_tokens}, recent: {recent_tokens})"
        )

        return {
            "summary": summary,
            "recent": recent_messages,
            "total_tokens": final_total,
            "compressed": True,
            "compression_stats": {
                "original_messages": len(history),
                "original_tokens": total_tokens,
                "summarized_messages": len(old_messages),
                "kept_messages": len(recent_messages),
                "final_tokens": final_total,
                "compression_ratio": f"{(1 - final_total/total_tokens)*100:.1f}%"
            }
        }

    async def _generate_summary(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate an academic-focused summary of old conversation messages.

        Extracts:
        - Papers discussed (titles, authors, years, findings)
        - Research topics and questions
        - User context (name, preferences)
        - Key metrics and results mentioned
        """
        # Format messages for LLM
        formatted_history = "\n\n".join([
            f"{msg.get('role', 'user').title()}: {msg.get('content', '')}"
            for msg in messages
        ])

        # Get prompts from config
        system_prompt = getattr(
            config.prompts,
            'history_summarizer_system',
            "You are a conversation summarizer for an academic research assistant. "
            "Extract key information: papers, authors, findings, user context."
        )

        user_prompt_template = getattr(
            config.prompts,
            'history_summarizer_user',
            "Conversation history to summarize:\n{history}\n\nGenerate a compact summary preserving academic context."
        )

        user_prompt = user_prompt_template.format(history=formatted_history)

        try:
            # Call LLM to generate summary
            summary = await llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Slightly creative but focused
                max_tokens=500  # Keep summary concise
            )

            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Fallback: simple truncation summary
            return f"Previous conversation covered {len(messages)} messages discussing various topics."


# Singleton instance
history_manager = HistoryManager()


# Convenience function for easy import
async def compress_history(
    history: List[Dict[str, Any]],
    max_tokens: int = None
) -> Dict[str, Any]:
    """
    Compress conversation history (convenience wrapper).

    Args:
        history: Conversation history
        max_tokens: Max tokens before compression (uses config default if None)

    Returns:
        Compressed history dict
    """
    return await history_manager.compress_history(history, max_tokens)
