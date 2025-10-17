"""Intent Router Agent - Fast-path classification for queries"""

import json
from typing import Dict, Any, List
from loguru import logger

from src.config import config
from src.core.llm_client import llm_client


class IntentRouterAgent:
    """
    Routes queries to fast-path or full pipeline based on intent.

    Intent Types:
    - greeting: "Hello", "Hi", "How are you?" → Instant response
    - meta_question: "What can you do?", "How do you work?" → Instant response
    - simple_info: "What papers do you have?" → Light retrieval
    - research_query: Complex questions → Full pipeline
    """

    def __init__(self):
        self.system_prompt = config.prompts.intent_router_system
        self.user_prompt_template = config.prompts.intent_router_user

    async def classify_intent(
        self,
        query: str,
        history: List[Dict[str, Any]] = None,
        history_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Classify user query intent for routing.

        Args:
            query: User query
            history: Recent conversation history
            history_summary: Summary of older conversation history

        Returns:
            Dict with:
                - intent: "greeting" | "meta_question" | "simple_info" | "research_query"
                - confidence: float (0.0-1.0)
                - reasoning: str
                - suggested_response: str (for greeting/meta only)
        """
        logger.info(f"🎯 Intent Router classifying: {query[:50]}...")

        if history is None:
            history = []

        # Format history (convert from role/content format)
        history_str = "\n".join([
            f"{h.get('role', 'user').title()}: {h.get('content', '')}"
            for h in history[-6:]  # Last 6 messages (3 turns)
        ]) if history else "No previous conversation"

        # Format summary
        summary_str = history_summary if history_summary else "No previous conversation summary"

        # Prepare prompt
        user_prompt = self.user_prompt_template.format(
            query=query,
            history_summary=summary_str,
            history=history_str
        )

        # Call LLM for intent classification
        response = await llm_client.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,  # Deterministic for consistent routing
            response_format={"type": "json_object"}
        )

        # Parse response
        result = json.loads(response)

        # Log decision
        intent = result.get('intent', 'research_query')
        confidence = result.get('confidence', 0.0)

        logger.info(
            f"📍 Intent: {intent} "
            f"(confidence: {confidence:.2f})"
        )

        return result


# Singleton instance
intent_router_agent = IntentRouterAgent()
