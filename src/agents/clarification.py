"""Clarification + Security Agent - LLM-based validation"""

import json
from typing import Dict, Any, List
from loguru import logger

from src.config import config
from src.core.llm_client import llm_client


class ClarificationAgent:
    """
    Validates queries for security threats and clarity issues.
    Uses semantic understanding, not keyword patterns.
    """

    def __init__(self):
        self.system_prompt = config.prompts.clarification_system
        self.user_prompt_template = config.prompts.clarification_user
        self.security_enabled = config.agents.enable_security_check

    async def validate(
        self,
        query: str,
        history: List[Dict[str, Any]] = None,
        history_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Validate query for security and clarity.

        Args:
            query: User query
            history: Recent conversation history
            history_summary: Summary of older conversation history

        Returns:
            Dict with:
                - is_safe: bool (false if security threat detected)
                - security_reasoning: str
                - needs_clarification: bool
                - clarity_issues: List[str]
                - clarification_questions: List[str]
                - recommended_action: "block" | "clarify" | "proceed"
        """
        logger.info(f"🔒 Validating query: {query[:100]}...")

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
            temperature=0.0,  # Low temperature for consistent security decisions
            response_format={"type": "json_object"}
        )

        # Parse response
        result = json.loads(response)

        # Log results
        if not result.get('is_safe', True):
            logger.warning(
                f"🚨 Security threat detected: {result.get('security_reasoning', 'Unknown')}"
            )
        elif result.get('needs_clarification', False):
            logger.info(
                f"❓ Clarification needed: {len(result.get('clarity_issues', []))} issues"
            )
        else:
            logger.success("✅ Query validated - proceeding")

        return result


# Singleton instance
clarification_agent = ClarificationAgent()
