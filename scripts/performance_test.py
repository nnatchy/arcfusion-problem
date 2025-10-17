"""Performance test for optimized system"""

import sys
from pathlib import Path
import asyncio
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.agents.orchestrator import orchestrator


async def test_greeting():
    """Test fast-path: greeting"""
    logger.info("=" * 80)
    logger.info("TEST 1: Greeting (Fast Path)")
    logger.info("=" * 80)

    query = "Hello! How are you?"

    start_time = time.time()
    result = await orchestrator.ask(query=query, session_id="perf_test_greeting")
    elapsed = time.time() - start_time

    logger.success(f"✅ Greeting completed in {elapsed:.2f}s")
    logger.info(f"Answer: {result.get('answer', '')[:100]}...")
    logger.info(f"Intent: {result.get('metadata', {}).get('intent', 'N/A')}")
    logger.info("")

    return elapsed


async def test_meta_question():
    """Test fast-path: meta question"""
    logger.info("=" * 80)
    logger.info("TEST 2: Meta Question (Fast Path)")
    logger.info("=" * 80)

    query = "What papers do you have?"

    start_time = time.time()
    result = await orchestrator.ask(query=query, session_id="perf_test_meta")
    elapsed = time.time() - start_time

    logger.success(f"✅ Meta question completed in {elapsed:.2f}s")
    logger.info(f"Answer: {result.get('answer', '')[:150]}...")
    logger.info("")

    return elapsed


async def test_simple_research():
    """Test full pipeline: simple research query"""
    logger.info("=" * 80)
    logger.info("TEST 3: Simple Research Query (Full Pipeline)")
    logger.info("=" * 80)

    query = "What is text-to-SQL?"

    start_time = time.time()
    result = await orchestrator.ask(query=query, session_id="perf_test_simple")
    elapsed = time.time() - start_time

    logger.success(f"✅ Simple research completed in {elapsed:.2f}s")
    logger.info(f"Answer preview: {result.get('answer', '')[:100]}...")
    logger.info(f"Sources: {len(result.get('sources', []))}")
    logger.info(f"Quality: {result.get('quality_score', 0.0):.2f}")
    logger.info(f"Iterations: {result.get('iterations', 0)}")
    logger.info("")

    return elapsed


async def test_complex_research():
    """Test full pipeline: complex research query"""
    logger.info("=" * 80)
    logger.info("TEST 4: Complex Research Query (Full Pipeline)")
    logger.info("=" * 80)

    query = "According to the papers, what are the main approaches to text-to-SQL and their accuracy?"

    start_time = time.time()
    result = await orchestrator.ask(query=query, session_id="perf_test_complex")
    elapsed = time.time() - start_time

    logger.success(f"✅ Complex research completed in {elapsed:.2f}s")
    logger.info(f"Answer preview: {result.get('answer', '')[:150]}...")
    logger.info(f"Sources: {len(result.get('sources', []))}")
    logger.info(f"Quality: {result.get('quality_score', 0.0):.2f}")
    logger.info(f"Iterations: {result.get('iterations', 0)}")
    logger.info(f"Routing: {result.get('routing_decision', 'N/A')}")
    logger.info("")

    return elapsed


async def main():
    """Run all performance tests"""
    logger.info("🚀 Starting Performance Tests")
    logger.info("Expected improvements:")
    logger.info("  - Greetings/Meta: ~300ms (fast-path)")
    logger.info("  - Simple queries: ~3-5s (optimized pipeline)")
    logger.info("  - Complex queries: ~8-10s (optimized)")
    logger.info("")

    try:
        # Test 1: Greeting
        t1 = await test_greeting()

        # Test 2: Meta question
        t2 = await test_meta_question()

        # Test 3: Simple research
        t3 = await test_simple_research()

        # Test 4: Complex research
        t4 = await test_complex_research()

        # Summary
        logger.info("=" * 80)
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Greeting (fast-path):      {t1:.2f}s (target: <1s)")
        logger.info(f"Meta question (fast-path): {t2:.2f}s (target: <1s)")
        logger.info(f"Simple research:           {t3:.2f}s (target: <5s)")
        logger.info(f"Complex research:          {t4:.2f}s (target: <10s)")
        logger.info("")

        # Check targets
        passed = 0
        failed = 0

        if t1 < 1.0:
            logger.success("✅ Greeting: PASSED")
            passed += 1
        else:
            logger.warning(f"⚠️ Greeting: FAILED (expected <1s, got {t1:.2f}s)")
            failed += 1

        if t2 < 1.0:
            logger.success("✅ Meta question: PASSED")
            passed += 1
        else:
            logger.warning(f"⚠️ Meta question: FAILED (expected <1s, got {t2:.2f}s)")
            failed += 1

        if t3 < 5.0:
            logger.success("✅ Simple research: PASSED")
            passed += 1
        else:
            logger.warning(f"⚠️ Simple research: FAILED (expected <5s, got {t3:.2f}s)")
            failed += 1

        if t4 < 10.0:
            logger.success("✅ Complex research: PASSED")
            passed += 1
        else:
            logger.warning(f"⚠️ Complex research: FAILED (expected <10s, got {t4:.2f}s)")
            failed += 1

        logger.info("")
        logger.info(f"Results: {passed}/4 passed, {failed}/4 failed")

        if passed == 4:
            logger.success("🎉 All performance targets met!")
        else:
            logger.warning(f"⚠️ {failed} tests did not meet targets")

    except Exception as e:
        logger.error(f"❌ Test suite failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
