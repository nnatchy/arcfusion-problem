"""
Test script to verify core components are working.
Tests: Config, Logging, LLM, Embeddings, Pinecone connection.
"""

import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.utils.logging_config import setup_logging
from src.config import config


async def test_config():
    """Test configuration loading"""
    logger.info("=" * 60)
    logger.info("🧪 Testing Configuration")
    logger.info("=" * 60)

    try:
        logger.info(f"✅ LLM Model: {config.llm.model}")
        logger.info(f"✅ Embedding Model: {config.embeddings.model}")
        logger.info(f"✅ Pinecone Index: {config.pinecone.index_name}")
        logger.info(f"✅ Vector Dimension: {config.pinecone.dimension}")
        logger.info(f"✅ Chunk Size: {config.rag.chunk_size}")
        logger.info(f"✅ Max Recursion Depth: {config.rag.max_recursion_depth}")
        logger.success("✅ Configuration loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def test_llm():
    """Test LLM client"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 Testing LLM Client")
    logger.info("=" * 60)

    try:
        from src.core.llm_client import llm_client

        response = await llm_client.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'Hello, ArcFusion!' in a friendly way.",
            temperature=0.7
        )

        logger.info(f"✅ LLM Response: {response}")
        logger.success("✅ LLM client working!")
        return True
    except Exception as e:
        logger.error(f"❌ LLM test failed: {e}")
        return False


async def test_embeddings():
    """Test embeddings client"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 Testing Embeddings Client")
    logger.info("=" * 60)

    try:
        from src.core.embeddings import embedding_client

        # Test single embedding
        text = "This is a test document about machine learning."
        embedding = await embedding_client.embed_text(text)

        logger.info(f"✅ Embedding dimension: {len(embedding)}")
        logger.info(f"✅ First 5 values: {embedding[:5]}")

        # Test batch embedding
        texts = [
            "First document about AI",
            "Second document about deep learning",
            "Third document about neural networks"
        ]
        embeddings = await embedding_client.embed_batch(texts)

        logger.info(f"✅ Batch embeddings: {len(embeddings)} vectors generated")
        logger.success("✅ Embeddings client working!")
        return True
    except Exception as e:
        logger.error(f"❌ Embeddings test failed: {e}")
        return False


async def test_pinecone():
    """Test Pinecone connection"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 Testing Pinecone Connection")
    logger.info("=" * 60)

    try:
        from src.retrieval.vector_store import get_vector_store

        vector_store = get_vector_store()

        # Get stats
        stats = vector_store.get_stats()
        logger.info(f"✅ Index Name: {stats.get('index_name')}")
        logger.info(f"✅ Total Vectors: {stats.get('total_vectors', 0)}")
        logger.info(f"✅ Dimension: {stats.get('dimension')}")

        logger.success("✅ Pinecone connection working!")
        return True
    except Exception as e:
        logger.error(f"❌ Pinecone test failed: {e}")
        return False


async def test_pdf_processor():
    """Test PDF processor (if PDFs exist)"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 Testing PDF Processor")
    logger.info("=" * 60)

    try:
        from src.retrieval.pdf_processor import pdf_processor

        papers_dir = Path("papers")
        if not papers_dir.exists() or not list(papers_dir.glob("*.pdf")):
            logger.warning("⚠️  No PDF files found in papers/ directory - skipping test")
            return True

        pdf_files = list(papers_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        # Test on first PDF
        test_pdf = pdf_files[0]
        logger.info(f"Testing with: {test_pdf.name}")

        # Extract metadata
        metadata = pdf_processor.extract_metadata(str(test_pdf))
        logger.info(f"✅ Metadata: {metadata.get('title')}")

        # Process PDF (just first page for testing)
        chunks = pdf_processor.process_pdf(str(test_pdf))
        logger.info(f"✅ Created {len(chunks)} chunks")
        logger.info(f"✅ First chunk preview: {chunks[0].text[:100]}...")

        logger.success("✅ PDF processor working!")
        return True
    except Exception as e:
        logger.error(f"❌ PDF processor test failed: {e}")
        return False


async def test_web_search():
    """Test Tavily web search"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 Testing Web Search (Tavily)")
    logger.info("=" * 60)

    try:
        from src.retrieval.web_searcher import web_searcher

        query = "latest developments in AI 2024"
        results = await web_searcher.search(query, max_results=3)

        logger.info(f"✅ Found {len(results)} results")
        if results:
            logger.info(f"✅ First result: {results[0].title}")
            logger.info(f"✅ URL: {results[0].url}")

        logger.success("✅ Web search working!")
        return True
    except Exception as e:
        logger.error(f"❌ Web search test failed: {e}")
        return False


async def main():
    """Run all tests"""
    # Setup logging
    setup_logging(config)

    logger.info("\n🚀 ArcFusion RAG System - Component Testing\n")

    results = {
        "Config": await test_config(),
        "LLM": await test_llm(),
        "Embeddings": await test_embeddings(),
        "Pinecone": await test_pinecone(),
        "PDF Processor": await test_pdf_processor(),
        "Web Search": await test_web_search()
    }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Summary")
    logger.info("=" * 60)

    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{component:20s} {status}")

    all_passed = all(results.values())

    if all_passed:
        logger.success("\n🎉 All tests passed! System is ready.")
    else:
        logger.warning("\n⚠️  Some tests failed. Please check the errors above.")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
