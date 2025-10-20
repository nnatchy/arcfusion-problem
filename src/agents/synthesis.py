"""Synthesis Agent - Combines information from multiple sources"""

import json
from typing import Dict, Any, List, Optional
from loguru import logger

from src.config import config
from src.core.llm_client import llm_client


class SynthesisAgent:
    """
    Combines RAG results and web search results into a coherent answer.
    Preserves citation chains and maintains academic rigor.
    """

    def __init__(self):
        self.system_prompt = config.prompts.synthesis_system
        self.user_prompt_template = config.prompts.synthesis_user

    async def synthesize(
        self,
        query: str,
        rag_results: Optional[Dict[str, Any]] = None,
        web_results: Optional[Dict[str, Any]] = None,
        history: List[Dict[str, Any]] = None,
        history_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Synthesize information from multiple sources.

        Args:
            query: Original user query
            rag_results: Results from Recursive RAG Agent
            web_results: Results from Web Search Agent
            history: Recent conversation history
            history_summary: Summary of older conversation history

        Returns:
            Dict with:
                - answer: str (synthesized answer)
                - sources: List of all sources used
                - citations: List of formatted citations
        """
        logger.info(f"🔄 Synthesizing results for: {query[:100]}...")

        if history is None:
            history = []

        # Format RAG results
        rag_str = self._format_rag_results(rag_results) if rag_results else "No PDF results."

        # Format web results
        web_str = self._format_web_results(web_results) if web_results else "No web results."

        # Format history (convert from role/content format)
        history_str = "\n".join([
            f"{h.get('role', 'user').title()}: {h.get('content', '')}"
            for h in history[-config.agents.synthesis_history_window:]  # Configurable history window
        ]) if history else "No previous conversation"

        # Format summary
        summary_str = history_summary if history_summary else "No previous conversation summary"

        # Prepare prompt
        user_prompt = self.user_prompt_template.format(
            query=query,
            rag_results=rag_str,
            web_results=web_str,
            history_summary=summary_str,
            history=history_str
        )

        # Call LLM to synthesize
        response = await llm_client.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=config.agents.synthesis_temperature
        )

        # Collect all sources
        sources = self._collect_sources(rag_results, web_results)

        logger.success(f"✅ Synthesized answer with {len(sources)} sources")

        return {
            "answer": response,
            "sources": sources,
            "num_rag_sources": len(rag_results.get('retrieval_results', [])) if rag_results else 0,
            "num_web_sources": len(web_results.get('search_results', [])) if web_results else 0
        }

    def _format_rag_results(self, rag_results: Dict[str, Any]) -> str:
        """Format RAG results with citation chains"""
        if not rag_results or not rag_results.get('retrieval_results'):
            return "No PDF results."

        retrieval = rag_results.get('retrieval_results', [])
        partial_answer = rag_results.get('partial_answer', '')
        depth = rag_results.get('depth', 0)

        formatted = [
            f"Retrieval Depth: {depth}",
            f"Partial Answer: {partial_answer}\n" if partial_answer else "",
            "Retrieved Documents:"
        ]

        for i, doc in enumerate(retrieval[:10], 1):  # Top 10
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            score = doc.get('score', 0.0)

            formatted.append(
                f"\n[PDF {i}] {metadata.get('source', 'Unknown')} "
                f"(Page {metadata.get('page', 'N/A')}, Score: {score:.3f})\n"
                f"{text[:400]}..."
            )

        return "\n".join(formatted)

    def _format_web_results(self, web_results: Dict[str, Any]) -> str:
        """Format web search results"""
        if not web_results or not web_results.get('search_results'):
            return "No web results."

        results = web_results.get('search_results', [])
        summary = web_results.get('summary', '')

        formatted = [
            f"Web Search Summary: {summary}\n" if summary else "",
            "Web Sources:"
        ]

        for i, result in enumerate(results, 1):
            formatted.append(
                f"\n[Web {i}] {result.title} (Score: {result.score:.3f})\n"
                f"URL: {result.url}\n"
                f"Date: {result.published_date or 'N/A'}\n"
                f"{result.content[:400]}..."
            )

        return "\n".join(formatted)

    def _collect_sources(
        self,
        rag_results: Optional[Dict[str, Any]],
        web_results: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collect all sources with metadata"""
        sources = []

        # Add RAG sources
        if rag_results and rag_results.get('retrieval_results'):
            for doc in rag_results['retrieval_results']:
                metadata = doc.get('metadata', {})
                sources.append({
                    "type": "pdf",
                    "source": metadata.get('source', 'Unknown'),
                    "page": metadata.get('page', 'N/A'),
                    "score": doc.get('score', 0.0),
                    "text_preview": doc.get('text', '')[:200]
                })

        # Add web sources
        if web_results and web_results.get('search_results'):
            for result in web_results['search_results']:
                sources.append({
                    "type": "web",
                    "title": result.title,
                    "url": result.url,
                    "date": result.published_date,
                    "score": result.score,
                    "text_preview": result.content[:200]
                })

        return sources


# Singleton instance
synthesis_agent = SynthesisAgent()
