"""
OpenAI embeddings client for text-embedding-ada-002.
Handles embedding generation with batching and error handling.
"""

from typing import List, Union
from openai import AsyncOpenAI
from loguru import logger
from src.config import config


class EmbeddingClient:
    """Async OpenAI embeddings client wrapper"""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.embeddings.model
        self.dimension = config.embeddings.dimension
        self.batch_size = config.embeddings.batch_size
        logger.info(f"✅ Embeddings client initialized (model={self.model}, dim={self.dimension})")

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (list of floats)
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )

            embedding = response.data[0].embedding

            logger.debug(f"Generated embedding: {len(embedding)} dimensions")

            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                logger.debug(f"Embedding batch {i // self.batch_size + 1}: {len(batch)} texts")

                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )

                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Batch complete: {len(batch_embeddings)} embeddings generated")

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                raise

        logger.info(f"✅ Generated {len(all_embeddings)} embeddings")

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Alias for embed_text for semantic clarity.

        Args:
            query: Search query

        Returns:
            Query embedding vector
        """
        return await self.embed_text(query)


# Global embeddings client instance
embedding_client = EmbeddingClient()
