"""Recursive RAG Agent - Multi-hop retrieval for academic papers"""

import json
import re
from typing import Dict, Any, List, Optional
from loguru import logger

from src.config import config
from src.core.llm_client import llm_client
from src.retrieval.recursive_retriever import get_recursive_retriever
from src.retrieval.vector_store import get_vector_store


class RecursiveRAGAgent:
    """
    Performs multi-hop retrieval following citation chains in academic papers.
    """

    def __init__(self):
        self.system_prompt = config.prompts.recursive_rag_system
        self.user_prompt_template = config.prompts.recursive_rag_user
        self.max_depth = config.rag.max_recursion_depth
        # Initialize vector store and retriever
        vector_store = get_vector_store()
        self.retriever = get_recursive_retriever(vector_store)

    async def retrieve(
        self,
        query: str,
        history: List[Dict[str, Any]] = None,
        depth: int = 0,
        parent_context: str = "",
        history_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Perform recursive retrieval with LLM-guided decisions.

        Args:
            query: User query or citation to retrieve
            history: Recent conversation history
            depth: Current recursion depth
            parent_context: Context from previous hops
            history_summary: Summary of older conversation history

        Returns:
            Dict with:
                - retrieval_results: List of retrieved documents with metadata
                - sufficient_info: bool
                - needs_more_retrieval: bool
                - citations_to_fetch: List[str]
                - partial_answer: str
                - confidence: float
                - depth: int (current depth)
        """
        logger.info(f"📚 Recursive RAG retrieval (depth {depth}/{self.max_depth})")

        if history is None:
            history = []

        # Extract paper filter from query if author/year is mentioned
        paper_filter = self._extract_paper_filter(query)

        # Perform retrieval using RecursiveRetriever
        retrieval_results = await self.retriever.retrieve(
            query=query,
            depth=depth,
            visited_ids=set(),
            metadata_filter=paper_filter
        )

        # Format retrieved context
        context_str = self._format_retrieval_results(retrieval_results)

        # Format history (convert from role/content format)
        history_str = "\n".join([
            f"{h.get('role', 'user').title()}: {h.get('content', '')}"
            for h in history[-6:]  # Last 6 messages
        ]) if history else "No previous conversation"

        # Format summary
        summary_str = history_summary if history_summary else "No previous conversation summary"

        # Prepare prompt for LLM decision
        user_prompt = self.user_prompt_template.format(
            query=query,
            depth=depth,
            max_depth=self.max_depth,
            context=context_str,
            parent_context=parent_context if parent_context else "None (initial retrieval)",
            history_summary=summary_str,
            history=history_str
        )

        # Call LLM to decide next steps
        response = await llm_client.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        # Parse response
        decision = json.loads(response)

        # Convert RetrievalNode objects to dicts for compatibility
        retrieval_dicts = []
        for node in retrieval_results:
            retrieval_dicts.append({
                'text': node.content,
                'metadata': node.metadata,
                'score': node.score,
                'id': node.node_id,
                'depth': node.depth,
                'citations': node.citations_extracted
            })

        # Add retrieval results to decision
        result = {
            **decision,
            "retrieval_results": retrieval_dicts,
            "depth": depth
        }

        # Log decision
        if decision.get('sufficient_info'):
            logger.success(
                f"✅ Sufficient information found at depth {depth} "
                f"(confidence: {decision.get('confidence', 0.0):.2f})"
            )
        elif decision.get('needs_more_retrieval') and depth < self.max_depth:
            citations = decision.get('citations_to_fetch', [])
            logger.info(f"🔄 Need more retrieval: {len(citations)} citations to follow")
        else:
            logger.info(f"🛑 Max depth reached or no more citations to follow")

        return result

    def _format_retrieval_results(self, results: List[Any]) -> str:
        """Format retrieval results for LLM prompt"""
        if not results:
            return "No documents retrieved."

        formatted = []
        for i, node in enumerate(results[:10], 1):  # Top 10
            # Handle both RetrievalNode objects and dicts
            if hasattr(node, 'metadata'):
                # It's a RetrievalNode object
                metadata = node.metadata
                text = node.content
                score = node.score
            else:
                # It's a dict
                metadata = node.get('metadata', {})
                text = node.get('text', '')
                score = node.get('score', 0.0)

            formatted.append(
                f"[Document {i}] (Score: {score:.3f})\n"
                f"Source: {metadata.get('source', 'Unknown')}\n"
                f"Page: {metadata.get('page', 'N/A')}\n"
                f"Text: {text[:500]}...\n"
            )

        return "\n".join(formatted)

    def _extract_paper_filter(self, query: str) -> Optional[Dict]:
        """
        Extract paper author/year from query for metadata filtering.

        This enables paper-aware retrieval when user mentions specific papers.

        Examples:
            - "Zhang et al. 2024" → filter for year "2024"
            - "Rajkumar et al. (2022)" → filter for year "2022"
            - "Chang and Fosler-Lussier 2023" → filter for year "2023"

        Args:
            query: User query to extract paper metadata from

        Returns:
            Pinecone metadata filter dict, or None if no paper mentioned

        Note:
            We filter by year only because:
            1. Pinecone serverless doesn't support $regex operator
            2. Title field already contains author names, so semantic search will prioritize correct paper
            3. Year is stored as string in Pinecone metadata
            4. This approach is simpler and more reliable than complex string matching
        """
        # Pattern 1: "Author et al. YYYY" or "Author et al. (YYYY)"
        pattern1 = r'([A-Z][a-z]+)\s+et\s+al\.?\s*\(?(\d{4})\)?'
        match = re.search(pattern1, query, re.IGNORECASE)

        if match:
            author = match.group(1)
            year = match.group(2)  # Keep as string
            logger.info(f"📄 Detected paper filter: {author} et al. {year}")

            # Filter by year only - semantic search will handle author matching
            return {"year": year}

        # Pattern 2: "Author and Author YYYY"
        pattern2 = r'([A-Z][a-z]+)\s+and\s+([A-Z][a-z-]+)\s+\(?(\d{4})\)?'
        match = re.search(pattern2, query)

        if match:
            author1 = match.group(1)
            author2 = match.group(2)
            year = match.group(3)  # Keep as string
            logger.info(f"📄 Detected paper filter: {author1} and {author2} {year}")

            # Filter by year only
            return {"year": year}

        # No paper mentioned
        return None


# Singleton instance
recursive_rag_agent = RecursiveRAGAgent()
