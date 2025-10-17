"""
BGE-reranker-v2-m3 for improving retrieval quality.
Reranks documents based on query relevance with semantic understanding.
"""

from typing import List, Dict, Tuple, Optional
from FlagEmbedding import FlagReranker
from loguru import logger
from src.config import config


class BGEReranker:
    """BGE reranker for improving retrieval quality"""

    def __init__(self):
        self.model_name = config.rag.reranker_model
        logger.info(f"Loading reranker model: {self.model_name}...")

        try:
            # Load model (cached after first load)
            # trust_remote_code=True is required for some models (e.g., gemma2-lightweight)
            self.reranker = FlagReranker(
                self.model_name,
                use_fp16=True,  # Faster inference with half precision
                trust_remote_code=True  # Allow custom model code
            )
            logger.success(f"✅ Reranker loaded: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on query relevance.

        Args:
            query: Search query
            documents: List of document texts
            top_n: Number of top results to return

        Returns:
            List of (original_index, relevance_score) tuples sorted by score
        """
        if not documents:
            return []

        top_n = top_n or config.rag.reranker_top_n

        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]

            # Get relevance scores
            scores = self.reranker.compute_score(pairs, normalize=True)

            # Handle single document case
            if not isinstance(scores, list):
                scores = [scores]

            # Create (index, score) pairs and sort by score descending
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top N
            result = indexed_scores[:top_n]

            logger.debug(f"Reranked {len(documents)} docs → top {len(result)}")
            if result:
                logger.debug(f"Score range: {result[0][1]:.4f} (best) to {result[-1][1]:.4f} (worst)")

            return result

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return original order with dummy scores
            return [(i, 1.0) for i in range(min(top_n, len(documents)))]

    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict],  # Each dict has 'text' and other fields
        top_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank documents with metadata preservation.

        Args:
            query: Search query
            documents: List of dicts with 'text' and other metadata
            top_n: Number of top results to return

        Returns:
            Reranked documents with rerank_score added
        """
        if not documents:
            return []

        # Extract text for reranking
        texts = [doc.get('text', '') for doc in documents]

        # Get reranked indices and scores
        reranked = self.rerank(query, texts, top_n)

        # Reconstruct documents with scores
        result = []
        for idx, score in reranked:
            doc = documents[idx].copy()
            doc['rerank_score'] = float(score)
            result.append(doc)

        logger.info(f"✅ Reranked {len(documents)} documents → top {len(result)}")

        return result


# Note: Reranker instance is created by vector_store when needed
