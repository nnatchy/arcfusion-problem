"""Planner Agent - Multi-step task decomposition"""

import json
from typing import Dict, Any, List
from loguru import logger

from src.config import config
from src.core.llm_client import llm_client


class PlannerAgent:
    """Breaks down complex queries into executable sub-tasks"""

    def __init__(self):
        self.system_prompt = config.prompts.planner_system
        self.user_prompt_template = config.prompts.planner_user

    async def plan(
        self,
        query: str,
        history: List[Dict[str, Any]] = None,
        history_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Create execution plan for the query.

        Args:
            query: User query
            history: Recent conversation history
            history_summary: Summary of older conversation history

        Returns:
            Dict with:
                - is_complex: bool
                - tasks: List of task objects
                - reasoning: Explanation of the plan
        """
        logger.info(f"📋 Planning query: {query[:100]}...")

        if history is None:
            history = []

        # Format history (convert from role/content format)
        history_str = "\n".join([
            f"{h.get('role', 'user').title()}: {h.get('content', '')}"
            for h in history[-6:]  # Last 6 messages
        ]) if history else "No previous conversation"

        # Format summary
        summary_str = history_summary if history_summary else "No previous conversation summary"

        # Prepare prompt
        user_prompt = self.user_prompt_template.format(
            query=query,
            history_summary=summary_str,
            history=history_str
        )

        # Call LLM with JSON response format
        response = await llm_client.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        # Parse response
        result = json.loads(response)

        if result.get('is_complex'):
            logger.info(
                f"🔀 Complex query detected: {len(result.get('tasks', []))} tasks planned"
            )
        else:
            logger.info("➡️ Single-step query")

        return result


# Singleton instance
planner_agent = PlannerAgent()
