"""Router Agent - Semantic routing for information sources"""

import json
from typing import Dict, Any
from loguru import logger

from src.config import config
from src.core.llm_client import llm_client


class RouterAgent:
    """Routes queries to appropriate information sources using semantic understanding"""

    def __init__(self):
        self.system_prompt = config.prompts.router_system
        self.user_prompt_template = config.prompts.router_user

    async def route(
        self,
        query: str,
        history: list[Dict[str, Any]] = None,
        context: str = "",
        history_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Determine routing decision for the query.

        Args:
            query: User query
            history: Recent conversation history
            context: Additional context
            history_summary: Summary of older conversation history

        Returns:
            Dict with:
                - reasoning: Explanation of decision
                - decision: "pdf_only" | "web_only" | "hybrid"
                - confidence: float 0.0-1.0
                - execution_order: "sequential" | "parallel" (for hybrid)
        """
        logger.info(f"🧭 Routing query: {query[:100]}...")

        if history is None:
            history = []

        # Format history (convert from role/content format)
        history_str = "\n".join([
            f"{h.get('role', 'user').title()}: {h.get('content', '')}"
            for h in history[-config.agents.router_history_window:]  # Configurable history window
        ]) if history else "No previous conversation"

        # Format summary
        summary_str = history_summary if history_summary else "No previous conversation summary"

        # Prepare prompt
        user_prompt = self.user_prompt_template.format(
            query=query,
            history_summary=summary_str,
            history=history_str,
            context=context if context else "No additional context"
        )

        # Call LLM with JSON response format
        response = await llm_client.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=config.agents.router_temperature,
            response_format={"type": "json_object"}
        )

        # Parse response
        result = json.loads(response)

        # Ensure execution_order is set for hybrid
        if result.get('decision') == 'hybrid' and 'execution_order' not in result:
            result['execution_order'] = 'parallel'  # Default to parallel

        logger.info(
            f"📍 Routing decision: {result['decision']} "
            f"(confidence: {result.get('confidence', 0.0):.2f})"
        )

        if result.get('decision') == 'hybrid':
            logger.info(f"⚡ Execution order: {result.get('execution_order', 'parallel')}")

        return result


# Singleton instance
router_agent = RouterAgent()
