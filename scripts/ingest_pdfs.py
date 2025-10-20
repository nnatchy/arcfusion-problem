"""
PDF ingestion script for ArcFusion RAG system.
Processes PDFs from papers/ directory and uploads to Pinecone.
"""

import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.utils.logging_config import setup_logging
from src.config import config
from src.retrieval.pdf_processor import pdf_processor
from src.retrieval.vector_store import get_vector_store
from src.core.embeddings import embedding_client


async def ingest_pdfs(pdf_directory: str = None):
    """
    Main ingestion function: process PDFs and upload to Pinecone.

    Args:
        pdf_directory: Directory containing PDF files (defaults to config.system.pdf_directory)
    """
    # Use config default if not specified
    if pdf_directory is None:
        pdf_directory = config.system.pdf_directory
    logger.info("=" * 60)
    logger.info("🚀 Starting PDF ingestion process")
    logger.info("=" * 60)

    # Get papers directory path
    papers_dir = Path(pdf_directory)
    if not papers_dir.exists():
        logger.error(f"Papers directory not found: {papers_dir}")
        logger.info("Creating papers/ directory...")
        papers_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("⚠️  Please add PDF files to papers/ directory and run again")
        return

    # Check for PDF files
    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"⚠️  No PDF files found in {papers_dir}")
        logger.info("Please add PDF files to papers/ directory and run again")
        return

    logger.info(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        logger.info(f"  - {pdf_file.name}")

    # Initialize vector store (creates index if needed)
    logger.info("\n📦 Initializing Pinecone vector store...")
    vector_store = get_vector_store()

    # Get current stats
    stats = vector_store.get_stats()
    logger.info(f"Current vectors in index: {stats.get('total_vectors', 0)}")

    # Process all PDFs
    logger.info("\n📄 Processing PDF files...")
    all_results = pdf_processor.process_directory(str(papers_dir))

    if not all_results:
        logger.error("❌ No PDFs were successfully processed")
        return

    # Collect all chunks
    all_chunks = []
    for pdf_name, chunks in all_results.items():
        logger.info(f"  {pdf_name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    logger.success(f"✅ Total chunks created: {len(all_chunks)}")

    # Generate embeddings
    logger.info("\n🔮 Generating embeddings...")
    texts = [chunk.text for chunk in all_chunks]

    try:
        embeddings = await embedding_client.embed_batch(texts)
        logger.success(f"✅ Generated {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        return

    # Prepare data for Pinecone
    logger.info("\n📤 Uploading to Pinecone...")
    ids = [chunk.chunk_id for chunk in all_chunks]
    metadata = []

    for chunk in all_chunks:
        meta = {
            **chunk.metadata,
            'text': chunk.text,  # Store text in metadata for retrieval
            'page': chunk.page
        }
        metadata.append(meta)

    # Upload to Pinecone
    try:
        vector_store.upsert(
            vectors=embeddings,
            ids=ids,
            metadata=metadata
        )
        logger.success(f"✅ Successfully uploaded {len(embeddings)} vectors to Pinecone")
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return

    # Final stats
    logger.info("\n📊 Final Statistics:")
    final_stats = vector_store.get_stats()
    logger.info(f"  Total vectors in index: {final_stats.get('total_vectors', 0)}")
    logger.info(f"  Index dimension: {final_stats.get('dimension', 0)}")
    logger.info(f"  Index name: {final_stats.get('index_name', 'unknown')}")

    logger.info("\n" + "=" * 60)
    logger.success("🎉 PDF ingestion completed successfully!")
    logger.info("=" * 60)


def main():
    """Entry point for ingestion script"""
    # Setup logging
    setup_logging(config)

    logger.info("ArcFusion RAG System - PDF Ingestion")
    logger.info(f"Config: {config.pinecone.index_name}")
    logger.info(f"Embedding model: {config.embeddings.model}")

    # Run ingestion
    try:
        asyncio.run(ingest_pdfs())
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
