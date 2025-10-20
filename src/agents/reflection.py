"""Reflection Agent - Quality control and answer refinement"""

import json
from typing import Dict, Any, List
from loguru import logger

from src.config import config
from src.core.llm_client import llm_client


class ReflectionAgent:
    """
    Reviews and refines generated answers for quality.
    Evaluates: accuracy, completeness, clarity, citations, relevance.
    """

    def __init__(self):
        self.system_prompt = config.prompts.reflection_system
        self.user_prompt_template = config.prompts.reflection_user
        self.quality_threshold = config.agents.reflection_quality_threshold

    async def reflect(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Review and refine the generated answer.

        Args:
            query: Original user query
            answer: Generated answer to review
            sources: List of sources used

        Returns:
            Dict with:
                - quality_score: float (0.0-1.0)
                - issues_found: List[str]
                - improvements_needed: List[str]
                - revised_answer: str | None (None if no revision needed)
                - formatting_improvements: str | None
        """
        logger.info(f"🪞 Reflecting on answer for: {query[:100]}...")

        # Format sources
        sources_str = self._format_sources(sources)

        # Prepare prompt
        user_prompt = self.user_prompt_template.format(
            query=query,
            answer=answer,
            sources=sources_str
        )

        # Call LLM to reflect
        response = await llm_client.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=config.agents.reflection_temperature,
            response_format={"type": "json_object"}
        )

        # Parse response
        result = json.loads(response)

        # Log reflection results
        quality_score = result.get('quality_score', 0.0)
        issues = result.get('issues_found', [])
        has_revision = result.get('revised_answer') is not None

        if quality_score >= self.quality_threshold:
            logger.success(f"✅ High quality answer (score: {quality_score:.2f}) - no revision needed")
        else:
            logger.warning(
                f"⚠️ Quality issues detected (score: {quality_score:.2f}): "
                f"{len(issues)} issues, revision {'provided' if has_revision else 'not provided'}"
            )

        return result

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for reflection prompt"""
        if not sources:
            return "No sources provided."

        formatted = []
        for i, source in enumerate(sources[:15], 1):  # Top 15 sources
            if source.get('type') == 'pdf':
                formatted.append(
                    f"[PDF {i}] {source.get('source', 'Unknown')} "
                    f"(Page {source.get('page', 'N/A')}, Score: {source.get('score', 0.0):.3f})"
                )
            elif source.get('type') == 'web':
                formatted.append(
                    f"[Web {i}] {source.get('title', 'Unknown')} "
                    f"({source.get('url', 'N/A')}, Score: {source.get('score', 0.0):.3f})"
                )

        return "\n".join(formatted)


# Singleton instance
reflection_agent = ReflectionAgent()
