"""
Tavily web search agent with full reference tracking.
Retrieves current information from the web with comprehensive metadata.
"""

from typing import List, Dict, Optional
from tavily import TavilyClient
from dataclasses import dataclass
from loguru import logger
from src.config import config


@dataclass
class WebSearchResult:
    """Structured web search result with full reference"""
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None
    raw_content: Optional[str] = None


class TavilySearchAgent:
    """
    Enhanced Tavily search agent with comprehensive reference tracking.
    Returns full references for citations.
    """

    def __init__(self):
        self.client = TavilyClient(api_key=config.tavily_api_key)
        self.max_results = config.web_search.max_results
        self.search_depth = config.web_search.search_depth
        logger.info(f"✅ Tavily search agent initialized (depth={self.search_depth})")

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_raw_content: bool = True,
        search_depth: Optional[str] = None
    ) -> List[WebSearchResult]:
        """
        Search the web using Tavily with full reference tracking.

        Args:
            query: Search query
            max_results: Number of results to return
            include_raw_content: Include full page content
            search_depth: 'basic' or 'advanced'

        Returns:
            List of WebSearchResult with full references
        """
        max_results = max_results or self.max_results
        search_depth = search_depth or self.search_depth

        logger.info(f"🔍 Searching web: '{query}' (depth={search_depth})")

        try:
            # Tavily search with context
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=include_raw_content,
                include_answer=True,
                include_images=False
            )

            results = []

            for item in response.get('results', []):
                result = WebSearchResult(
                    title=item.get('title', 'Unknown'),
                    url=item.get('url', ''),
                    content=item.get('content', ''),
                    score=item.get('score', 0.0),
                    published_date=item.get('published_date'),
                    raw_content=item.get('raw_content') if include_raw_content else None
                )
                results.append(result)

            logger.success(f"✅ Found {len(results)} web results")

            # Log Tavily's AI answer if available
            if 'answer' in response:
                logger.debug(f"Tavily AI summary: {response['answer'][:150]}...")

            return results

        except Exception as e:
            logger.error(f"❌ Tavily search failed: {e}")
            return []

    def format_references(self, results: List[WebSearchResult]) -> List[Dict]:
        """
        Format search results as proper references for citation.

        Args:
            results: List of search results

        Returns:
            List of reference dicts
        """
        references = []

        for i, result in enumerate(results, 1):
            references.append({
                'type': 'web',
                'reference_id': f'web_{i}',
                'title': result.title,
                'url': result.url,
                'excerpt': result.content[:300] + "..." if len(result.content) > 300 else result.content,
                'published_date': result.published_date,
                'relevance_score': result.score
            })

        return references

    async def search_with_context(
        self,
        query: str,
        conversation_context: Optional[str] = None
    ) -> Dict:
        """
        Search with conversation context for better results.

        Args:
            query: Search query
            conversation_context: Previous conversation context

        Returns:
            Dict with results and formatted references
        """
        # Enhance query with context if available
        enhanced_query = query
        if conversation_context:
            enhanced_query = f"{conversation_context}\n\nQuery: {query}"

        results = await self.search(enhanced_query)
        references = self.format_references(results)

        return {
            'results': results,
            'references': references,
            'query_used': query,  # Use original query, not enhanced
            'total_results': len(results)
        }


# Global instance
web_searcher = TavilySearchAgent()
