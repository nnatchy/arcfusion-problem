"""Web Search Agent - Tavily integration for current information"""

import json
from typing import Dict, Any, List
from loguru import logger

from src.config import config
from src.core.llm_client import llm_client
from src.retrieval.web_searcher import web_searcher


class WebSearchAgent:
    """
    Performs web search using Tavily and summarizes findings.
    """

    def __init__(self):
        self.system_prompt = config.prompts.web_search_system
        self.user_prompt_template = config.prompts.web_search_user
        self.searcher = web_searcher
        self.max_results = config.web_search.max_results

    async def search(
        self,
        query: str,
        max_results: int = None
    ) -> Dict[str, Any]:
        """
        Search the web and provide summarized answer.

        Args:
            query: Search query
            max_results: Maximum number of results (default from config)

        Returns:
            Dict with:
                - search_results: List of WebSearchResult objects
                - summary: str (LLM-generated summary)
                - reformulated_query: str (query used for search)
        """
        if max_results is None:
            max_results = self.max_results

        logger.info(f"🌐 Web search: {query[:100]}...")

        # Perform web search
        search_results = await self.searcher.search(
            query=query,
            max_results=max_results
        )

        logger.info(f"📊 Found {len(search_results)} web results")

        # Format results for LLM
        results_str = self._format_search_results(search_results)

        # Prepare prompt
        user_prompt = self.user_prompt_template.format(
            query=query,
            results=results_str
        )

        # Call LLM to summarize findings
        response = await llm_client.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=config.agents.web_search_temperature
        )

        logger.success("✅ Web search summary generated")

        return {
            "search_results": search_results,
            "summary": response,
            "reformulated_query": query,
            "num_results": len(search_results)
        }

    def _format_search_results(self, results: List[Any]) -> str:
        """Format search results for LLM prompt"""
        if not results:
            return "No search results found."

        formatted = []
        for i, result in enumerate(results, 1):
            # WebSearchResult has: title, url, content, score, published_date
            formatted.append(
                f"[Result {i}] (Score: {result.score:.3f})\n"
                f"Title: {result.title}\n"
                f"URL: {result.url}\n"
                f"Date: {result.published_date or 'N/A'}\n"
                f"Content: {result.content[:500]}...\n"
            )

        return "\n".join(formatted)


# Singleton instance
web_search_agent = WebSearchAgent()
