"""Streamlit playground for ArcFusion RAG System"""

import streamlit as st
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config
from src.utils.logging_config import setup_logging
from src.agents.orchestrator import orchestrator

# Setup
setup_logging(config)

# Page config
st.set_page_config(
    page_title=config.streamlit.title,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_session"


def format_sources(sources):
    """Format sources for display"""
    if not sources:
        return "No sources"

    formatted = []
    for i, source in enumerate(sources, 1):
        if source.get('type') == 'pdf':
            formatted.append(
                f"**[{i}] PDF:** {source.get('source', 'Unknown')} "
                f"(Page {source.get('page', 'N/A')}, Score: {source.get('score', 0.0):.3f})"
            )
        elif source.get('type') == 'web':
            url = source.get('url', '#')
            formatted.append(
                f"**[{i}] Web:** [{source.get('title', 'Unknown')}]({url}) "
                f"(Score: {source.get('score', 0.0):.3f})"
            )

    return "\n".join(formatted)


async def process_query(query: str):
    """Process user query through orchestrator"""
    result = await orchestrator.ask(
        query=query,
        session_id=st.session_state.session_id
    )
    return result


# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    # Session management
    st.subheader("Session")
    st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        key="session_id_input",
        disabled=True
    )

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        orchestrator.clear_memory(st.session_state.session_id)
        st.success("History cleared!")
        st.rerun()

    st.divider()

    # System info
    st.subheader("System Info")
    st.text(f"LLM: {config.llm.model}")
    st.text(f"Embeddings: {config.embeddings.model}")
    st.text(f"Vector Store: {config.pinecone.index_name}")
    st.text(f"Max Recursion: {config.rag.max_recursion_depth}")

    st.divider()

    # Test queries
    st.subheader("📝 Example Queries")

    example_queries = [
        "What approaches were discussed in the text-to-SQL papers?",
        "Compare the performance metrics across different models",
        "What are the latest developments in text-to-SQL?",
        "Summarize the Zhang et al. paper findings"
    ]

    for query in example_queries:
        if st.button(f"📌 {query[:40]}...", key=query, use_container_width=True):
            st.session_state.test_query = query

# Main content
st.title("🧠 ArcFusion RAG Playground")
st.markdown("**Multi-agent RAG system with recursive retrieval and web search**")

st.divider()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("📊 Details"):
                metadata = message["metadata"]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", metadata.get('status', 'unknown'))
                with col2:
                    st.metric("Quality Score", f"{metadata.get('quality_score', 0.0):.2f}")
                with col3:
                    st.metric("Routing", metadata.get('routing_decision', 'N/A'))

                # Sources
                if metadata.get('sources'):
                    st.markdown("**📚 Sources:**")
                    st.markdown(format_sources(metadata['sources']))

                # Additional info
                if metadata.get('iterations'):
                    st.info(f"🔄 Iterations: {metadata['iterations']}")

# Chat input
if prompt := st.chat_input("Ask a question about academic papers..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            result = asyncio.run(process_query(prompt))

        status = result.get('status', 'error')

        if status == 'success':
            answer = result.get('answer', 'No answer generated')
            st.markdown(answer)

            # Store assistant message with metadata
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "metadata": {
                    "status": status,
                    "quality_score": result.get('quality_score', 0.0),
                    "routing_decision": result.get('routing_decision'),
                    "iterations": result.get('iterations', 0),
                    "sources": result.get('sources', [])
                }
            })

            # Show details
            with st.expander("📊 Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", "✅ Success")
                with col2:
                    st.metric("Quality Score", f"{result.get('quality_score', 0.0):.2f}")
                with col3:
                    st.metric("Routing", result.get('routing_decision', 'N/A'))

                # Sources
                sources = result.get('sources', [])
                if sources:
                    st.markdown("**📚 Sources:**")
                    st.markdown(format_sources(sources))

                # Additional info
                if result.get('iterations'):
                    st.info(f"🔄 Reflection iterations: {result['iterations']}")

        elif status == 'blocked':
            reason = result.get('reason', 'Security threat detected')
            st.error(f"🚨 **Query Blocked**\n\n{reason}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🚨 Query blocked: {reason}",
                "metadata": {"status": "blocked"}
            })

        elif status == 'clarification_needed':
            questions = result.get('questions', [])
            issues = result.get('issues', [])

            st.warning("❓ **Clarification Needed**")

            if issues:
                st.markdown("**Issues:**")
                for issue in issues:
                    st.markdown(f"- {issue}")

            if questions:
                st.markdown("**Questions:**")
                for question in questions:
                    st.markdown(f"- {question}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": "❓ I need clarification. See details above.",
                "metadata": {
                    "status": "clarification_needed",
                    "questions": questions,
                    "issues": issues
                }
            })

        elif status == 'error':
            error = result.get('error', 'Unknown error')
            st.error(f"❌ **Error**\n\n{error}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {error}",
                "metadata": {"status": "error"}
            })

# Handle test query button clicks - process immediately
if "test_query" in st.session_state and st.session_state.test_query:
    prompt = st.session_state.test_query
    st.session_state.test_query = None  # Clear it

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            result = asyncio.run(process_query(prompt))

        status = result.get('status', 'error')

        if status == 'success':
            answer = result.get('answer', 'No answer generated')
            st.markdown(answer)

            # Store assistant message with metadata
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "metadata": {
                    "status": status,
                    "quality_score": result.get('quality_score', 0.0),
                    "routing_decision": result.get('routing_decision'),
                    "iterations": result.get('iterations', 0),
                    "sources": result.get('sources', [])
                }
            })

            # Show details
            with st.expander("📊 Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", "✅ Success")
                with col2:
                    st.metric("Quality Score", f"{result.get('quality_score', 0.0):.2f}")
                with col3:
                    st.metric("Routing", result.get('routing_decision', 'N/A'))

                # Sources
                sources = result.get('sources', [])
                if sources:
                    st.markdown("**📚 Sources:**")
                    st.markdown(format_sources(sources))

                # Additional info
                if result.get('iterations'):
                    st.info(f"🔄 Reflection iterations: {result['iterations']}")

        elif status == 'blocked':
            reason = result.get('reason', 'Security threat detected')
            st.error(f"🚨 **Query Blocked**\n\n{reason}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🚨 Query blocked: {reason}",
                "metadata": {"status": "blocked"}
            })

        elif status == 'clarification_needed':
            questions = result.get('questions', [])
            issues = result.get('issues', [])

            st.warning("❓ **Clarification Needed**")

            if issues:
                st.markdown("**Issues:**")
                for issue in issues:
                    st.markdown(f"- {issue}")

            if questions:
                st.markdown("**Questions:**")
                for question in questions:
                    st.markdown(f"- {question}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": "❓ I need clarification. See details above.",
                "metadata": {
                    "status": "clarification_needed",
                    "questions": questions,
                    "issues": issues
                }
            })

        elif status == 'error':
            error = result.get('error', 'Unknown error')
            st.error(f"❌ **Error**\n\n{error}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {error}",
                "metadata": {"status": "error"}
            })

    st.rerun()

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🧠 ArcFusion RAG System | "
    "Powered by LangGraph, OpenAI, Pinecone, Tavily"
    "</div>",
    unsafe_allow_html=True
)
