"""Simple test for one specific query about the papers"""

import sys
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.config import config

setup_logging(config)

from loguru import logger
from src.agents.orchestrator import orchestrator


async def simple_test():
    """Test with a specific query about Zhang et al. paper"""
    logger.info("=" * 80)
    logger.info("🧪 Simple Test - Asking about Zhang et al. 2024 paper")
    logger.info("=" * 80)

    query = "According to the Zhang et al. 2024 paper, what were the main findings about GPT-4's text-to-SQL performance?"

    logger.info(f"\nQuery: {query}\n")

    result = await orchestrator.ask(
        query=query,
        session_id="simple_test_session"
    )

    logger.info("\n" + "=" * 80)
    logger.info("📊 RESULT")
    logger.info("=" * 80)
    logger.info(f"Status: {result.get('status')}")

    if result.get('status') == 'success':
        logger.success(f"\n✅ Answer:\n{result.get('answer', 'No answer')}\n")
        logger.info(f"Quality Score: {result.get('quality_score', 0.0):.2f}")
        logger.info(f"Routing: {result.get('routing_decision')}")
        logger.info(f"Sources: {result.get('metadata', {}).get('num_sources', 0)}")
        logger.info(f"Iterations: {result.get('iterations', 0)}")

        # Show sources
        sources = result.get('sources', [])
        if sources:
            logger.info(f"\n📚 Sources ({len(sources)} total):")
            for i, source in enumerate(sources[:5], 1):  # Show first 5
                if source.get('type') == 'pdf':
                    logger.info(f"  {i}. {source.get('source', 'Unknown')} (Page {source.get('page', 'N/A')}, Score: {source.get('score', 0.0):.3f})")
                elif source.get('type') == 'web':
                    logger.info(f"  {i}. {source.get('title', 'Unknown')} ({source.get('url', 'N/A')}, Score: {source.get('score', 0.0):.3f})")

    elif result.get('status') == 'blocked':
        logger.warning(f"🚨 Blocked: {result.get('reason', 'Unknown')}")
    elif result.get('status') == 'clarification_needed':
        logger.info(f"❓ Clarification needed:")
        for q in result.get('questions', []):
            logger.info(f"  - {q}")
    elif result.get('status') == 'error':
        logger.error(f"❌ Error: {result.get('error', 'Unknown')}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(simple_test())
