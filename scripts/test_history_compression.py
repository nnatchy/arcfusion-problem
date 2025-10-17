"""
Test token-based history compression with the orchestrator.

This script simulates a conversation that exceeds the token limit
to verify that history compression works correctly.
"""

import asyncio
from src.agents.orchestrator import orchestrator


async def test_history_compression():
    """Test history compression with a long conversation"""

    print("=" * 80)
    print("🧪 Testing Token-Based History Compression")
    print("=" * 80)

    session_id = "test_compression_session"

    # Create a series of queries that will generate long responses
    queries = [
        "Hi, my name is Alice and I'm researching text-to-SQL models.",
        "What papers do you have about text-to-SQL?",
        "Tell me about the accuracy results in those papers.",
        "What methods did the authors use?",
        "Compare the different approaches mentioned.",
        "What datasets were used for evaluation?",
        "What are the main findings?",
        "Are there any limitations discussed?",
        "What future work do the authors suggest?",
        "Can you summarize the key contributions?",
        # These should trigger compression if responses are long
        "What is my name?",  # Test memory after potential compression
        "What was the first paper I asked about?",  # Test if old context is preserved
    ]

    print(f"\n📝 Running {len(queries)} queries to test compression...")
    print(f"📊 Compression threshold: 10,000 tokens")
    print(f"🔄 Recent history window: 5,000 tokens\n")

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(queries)}: {query}")
        print(f"{'='*80}")

        result = await orchestrator.ask(query, session_id=session_id)

        if result.get('status') == 'success':
            answer = result.get('answer', 'No answer')
            print(f"\n✅ Answer ({len(answer)} chars):")
            print(f"{answer[:500]}...")  # Show first 500 chars

            # Show compression info if available from logs
            print(f"\n📊 Metadata:")
            print(f"  - Quality Score: {result.get('quality_score', 0.0):.2f}")
            print(f"  - Sources: {result.get('metadata', {}).get('num_sources', 0)}")
            print(f"  - Routing: {result.get('routing_decision', 'N/A')}")
        else:
            print(f"\n❌ Status: {result.get('status')}")
            if result.get('error'):
                print(f"Error: {result.get('error')}")

    print("\n" + "=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)
    print("\nCheck the logs above for:")
    print("  1. '📊 History stats' - shows message and token counts")
    print("  2. '🗜️ History compressed' - appears when compression is triggered")
    print("  3. Memory recall - verify name and first paper are remembered")


if __name__ == "__main__":
    asyncio.run(test_history_compression())
