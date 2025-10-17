"""
Pinecone vector store with automatic index creation.
Handles vector storage, retrieval, and reranking.
"""

from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec
from loguru import logger
from src.config import config
import time


class PineconeVectorStore:
    """Pinecone vector database interface with auto-creation"""

    def __init__(self):
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self.index_name = config.pinecone.index_name
        self.dimension = config.pinecone.dimension

        # Auto-create index if doesn't exist
        self._initialize_index()

        # Connect to index
        self.index = self.pc.Index(self.index_name)
        logger.success(f"✅ Connected to Pinecone index: {self.index_name}")

        # Initialize reranker if enabled
        self.reranker = None
        if config.rag.use_reranking:
            try:
                from src.retrieval.reranker import BGEReranker
                self.reranker = BGEReranker()
                logger.info("✅ Reranker enabled")
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                logger.warning("⚠️  Continuing without reranking (will use vector similarity only)")
                self.reranker = None

    def _initialize_index(self):
        """
        Automatically create Pinecone index if it doesn't exist.
        This runs on first startup.
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(f"Index '{self.index_name}' not found, creating...")

            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=config.pinecone.metric,
                spec=ServerlessSpec(
                    cloud=config.pinecone.cloud,
                    region=config.pinecone.region
                )
            )

            logger.success(f"✅ Created Pinecone index: {self.index_name}")
            logger.info(f"   Dimensions: {self.dimension}")
            logger.info(f"   Metric: {config.pinecone.metric}")
            logger.info(f"   Cloud: {config.pinecone.cloud}/{config.pinecone.region}")

            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            time.sleep(10)  # Serverless index takes ~10 seconds to initialize

        else:
            logger.info(f"Index '{self.index_name}' already exists")

    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict]
    ):
        """
        Insert or update vectors with metadata.

        Args:
            vectors: List of embedding vectors
            ids: List of unique IDs for each vector
            metadata: List of metadata dicts for each vector
        """
        if not vectors or not ids or not metadata:
            logger.warning("Empty vectors, ids, or metadata provided to upsert")
            return

        if not (len(vectors) == len(ids) == len(metadata)):
            raise ValueError("vectors, ids, and metadata must have the same length")

        # Prepare vectors with metadata
        vectors_with_metadata = [
            {
                "id": id_,
                "values": vec,
                "metadata": meta
            }
            for id_, vec, meta in zip(ids, vectors, metadata)
        ]

        # Batch upsert
        batch_size = 100
        total_upserted = 0

        for i in range(0, len(vectors_with_metadata), batch_size):
            batch = vectors_with_metadata[i:i + batch_size]

            try:
                self.index.upsert(
                    vectors=batch,
                    namespace=config.pinecone.namespace
                )
                total_upserted += len(batch)
                logger.debug(f"Upserted batch: {len(batch)} vectors")

            except Exception as e:
                logger.error(f"Batch upsert failed: {e}")
                raise

        logger.info(f"✅ Upserted {total_upserted} vectors to Pinecone")

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Query similar vectors.

        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Include metadata in results

        Returns:
            List of matching documents with scores and metadata
        """
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter,
                namespace=config.pinecone.namespace,
                include_metadata=include_metadata
            )

            documents = [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata if include_metadata else {},
                    "text": match.metadata.get("text", "") if include_metadata else ""
                }
                for match in results.matches
            ]

            logger.debug(f"Retrieved {len(documents)} documents from Pinecone")

            return documents

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def query_with_reranking(
        self,
        query: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query with optional reranking for better results.

        Args:
            query: Original text query (for reranking)
            query_vector: Query embedding
            top_k: Final number of results to return
            filter: Optional metadata filter

        Returns:
            List of reranked documents
        """
        # Initial retrieval with higher top_k
        initial_k = config.rag.top_k_retrieval
        results = self.query(
            query_vector=query_vector,
            top_k=initial_k,
            filter=filter
        )

        logger.info(f"Retrieved {len(results)} initial candidates")

        # Rerank if enabled
        if self.reranker and results:
            documents = [
                {
                    'text': r.get('text', ''),
                    'metadata': r.get('metadata', {}),
                    'id': r.get('id'),
                    'initial_score': r.get('score')
                }
                for r in results
            ]

            reranked = self.reranker.rerank_with_metadata(
                query=query,
                documents=documents,
                top_n=top_k or config.rag.top_k_final
            )

            logger.info(f"✅ Reranked to top {len(reranked)} results")
            return reranked

        # Return top_k without reranking
        final_k = top_k or config.rag.top_k_final
        return results[:final_k]

    def get_stats(self) -> Dict:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_name": self.index_name
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def delete_all(self):
        """Delete all vectors (use with caution)"""
        logger.warning(f"Deleting all vectors from index '{self.index_name}'")
        self.index.delete(delete_all=True, namespace=config.pinecone.namespace)
        logger.warning("✅ All vectors deleted")


# Global vector store instance will be initialized when needed
vector_store = None


def get_vector_store() -> PineconeVectorStore:
    """Get or create global vector store instance"""
    global vector_store
    if vector_store is None:
        vector_store = PineconeVectorStore()
    return vector_store
