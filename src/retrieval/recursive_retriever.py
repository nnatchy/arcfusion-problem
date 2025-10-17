"""
Recursive retriever for following citation chains in academic papers.
Performs multi-hop retrieval to answer complex questions.
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from loguru import logger
from src.config import config
from src.core.embeddings import embedding_client
import json
import re


@dataclass
class RetrievalNode:
    """Represents a node in the retrieval tree"""
    content: str
    metadata: Dict
    depth: int
    parent_id: Optional[str] = None
    citations_extracted: List[str] = field(default_factory=list)
    score: float = 0.0
    node_id: str = ""


class RecursiveRetriever:
    """
    Recursive retriever that follows citation chains and cross-references.
    Perfect for academic papers with complex citation networks.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.max_depth = config.rag.max_recursion_depth
        self.follow_citations = config.rag.follow_citations
        logger.info(f"✅ Recursive retriever initialized (max_depth={self.max_depth})")

    async def retrieve(
        self,
        query: str,
        initial_top_k: Optional[int] = None,
        depth: int = 0,
        visited_ids: Optional[Set[str]] = None
    ) -> List[RetrievalNode]:
        """
        Recursively retrieve documents following citation chains.

        Args:
            query: Search query
            initial_top_k: Number of initial results
            depth: Current recursion depth
            visited_ids: Set of already visited document IDs (prevent cycles)

        Returns:
            List of RetrievalNode objects with citation chains
        """
        if visited_ids is None:
            visited_ids = set()

        if depth >= self.max_depth:
            logger.debug(f"Max recursion depth {self.max_depth} reached")
            return []

        logger.info(f"🔍 Recursive retrieval at depth {depth} for: {query[:80]}...")

        # Generate query embedding
        query_embedding = await embedding_client.embed_query(query)

        # Initial retrieval
        top_k = initial_top_k or config.rag.top_k_retrieval
        results = self.vector_store.query_with_reranking(
            query=query,
            query_vector=query_embedding,
            top_k=top_k
        )

        nodes = []

        for result in results:
            doc_id = result.get('id')

            # Skip if already visited (prevent cycles)
            if doc_id in visited_ids:
                logger.debug(f"Skipping already visited doc: {doc_id}")
                continue

            visited_ids.add(doc_id)

            # Create node
            node = RetrievalNode(
                content=result.get('text', ''),
                metadata=result.get('metadata', {}),
                depth=depth,
                score=result.get('rerank_score', result.get('initial_score', 0.0)),
                node_id=doc_id
            )

            # Extract citations if enabled
            if self.follow_citations and config.rag.citation_extraction_enabled:
                citations = self._extract_citations_regex(node.content)
                node.citations_extracted = citations

                if citations:
                    logger.debug(f"📚 Found {len(citations)} citations in doc {doc_id}")

                # Recursive retrieval for top citations (limit to 2 per doc)
                for citation in citations[:2]:
                    logger.info(f"  ↳ Following citation at depth {depth + 1}: {citation}")

                    child_nodes = await self.retrieve(
                        query=citation,
                        initial_top_k=3,  # Fewer results for nested retrievals
                        depth=depth + 1,
                        visited_ids=visited_ids
                    )

                    # Link child nodes to parent
                    for child in child_nodes:
                        child.parent_id = doc_id

                    nodes.extend(child_nodes)

            nodes.append(node)

        logger.info(f"✅ Retrieved {len(nodes)} nodes at depth {depth}")
        return nodes

    def _extract_citations_regex(self, text: str) -> List[str]:
        """
        Extract citations from text using regex patterns.
        Fallback method for quick extraction.

        Args:
            text: Text to extract citations from

        Returns:
            List of citation strings
        """
        # Patterns for common citation formats
        patterns = [
            r'\b([A-Z][a-z]+\s+et\s+al\.\s*,?\s*\(?\d{4}\)?)',
            r'\b([A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s*,?\s*\(?\d{4}\)?)',
            r'\b([A-Z][a-z]+\s+&\s+[A-Z][a-z]+\s*,?\s*\(?\d{4}\)?)',
            r'\[([A-Z][a-z]+\s+et\s+al\.\s*,?\s*\d{4})\]'
        ]

        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text[:1000])  # Check first 1000 chars
            citations.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            citation_clean = citation.strip()
            if citation_clean and citation_clean not in seen:
                seen.add(citation_clean)
                unique_citations.append(citation_clean)

        return unique_citations[:5]  # Max 5 citations

    def format_results_with_provenance(
        self,
        nodes: List[RetrievalNode]
    ) -> Dict:
        """
        Format retrieval results with full citation provenance.
        Shows the retrieval tree structure.

        Args:
            nodes: List of retrieval nodes

        Returns:
            Formatted dict with retrieval tree
        """
        if not nodes:
            return {
                "total_nodes": 0,
                "max_depth_reached": 0,
                "retrieval_tree": []
            }

        # Group by depth for clarity
        by_depth = {}
        for node in nodes:
            if node.depth not in by_depth:
                by_depth[node.depth] = []
            by_depth[node.depth].append(node)

        formatted = {
            "total_nodes": len(nodes),
            "max_depth_reached": max([n.depth for n in nodes]),
            "retrieval_tree": []
        }

        for depth in sorted(by_depth.keys()):
            depth_nodes = []
            for n in by_depth[depth]:
                depth_nodes.append({
                    "content_preview": n.content[:200] + "..." if len(n.content) > 200 else n.content,
                    "paper": n.metadata.get("title", "Unknown"),
                    "authors": n.metadata.get("author", "Unknown"),
                    "page": n.metadata.get("page", "N/A"),
                    "score": round(n.score, 3),
                    "citations_found": n.citations_extracted,
                    "parent_id": n.parent_id,
                    "node_id": n.node_id
                })

            formatted["retrieval_tree"].append({
                "depth": depth,
                "node_count": len(depth_nodes),
                "nodes": depth_nodes
            })

        return formatted

    async def retrieve_with_context(
        self,
        query: str,
        conversation_history: Optional[List] = None
    ) -> Dict:
        """
        Intelligent recursive retrieval with context awareness.

        Args:
            query: Search query
            conversation_history: Previous conversation turns

        Returns:
            Dict with nodes and formatted context
        """
        logger.info(f"🔍 Starting recursive retrieval for: {query}")

        # Initial retrieval
        initial_nodes = await self.retrieve(query, depth=0)

        # Format results
        provenance = self.format_results_with_provenance(initial_nodes)

        # Build final context for LLM
        final_context = []
        for node in initial_nodes:
            final_context.append({
                "text": node.content,
                "metadata": node.metadata,
                "depth": node.depth,
                "score": node.score,
                "citations": node.citations_extracted,
                "parent_id": node.parent_id
            })

        result = {
            "nodes": initial_nodes,
            "context": final_context,
            "provenance": provenance,
            "total_retrieved": len(initial_nodes)
        }

        logger.success(f"✅ Recursive retrieval complete: {len(initial_nodes)} nodes retrieved")

        return result


# Global instance will be created when needed
recursive_retriever = None


def get_recursive_retriever(vector_store):
    """Get or create global recursive retriever instance"""
    global recursive_retriever
    if recursive_retriever is None:
        recursive_retriever = RecursiveRetriever(vector_store)
    return recursive_retriever
