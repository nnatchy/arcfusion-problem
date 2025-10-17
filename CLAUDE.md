# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ArcFusion is a **multi-agent RAG system** for academic paper Q&A using LangGraph. It features recursive retrieval (follows citation chains), LLM-based security validation, web search integration, and a reflection loop for quality control. All logic is config-driven with **zero hardcoded decision rules**.

**Tech Stack**: Python 3.11, LangGraph, OpenAI (GPT-4o-mini), Pinecone, Tavily, FastAPI, Streamlit, uv package manager

## Development Commands

```bash
# Install dependencies (uses uv - 10x faster than pip)
uv sync

# Test core components
uv run python scripts/quick_test.py

# Ingest PDFs to Pinecone
uv run python scripts/ingest_pdfs.py

# Test end-to-end agent system
uv run python scripts/simple_test.py

# Test token-based history compression
uv run python scripts/test_history_compression.py

# Run FastAPI server
./scripts/run_api.sh
# or: uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit UI
./scripts/run_streamlit.sh
# or: uv run streamlit run src/ui/streamlit_app.py --server.port 8501

# View logs
tail -f logs/app.log
```

## Architecture Overview

### Multi-Agent Pipeline (LangGraph StateGraph)

```
User Query → [Intent Router] → Fast path (greetings/meta) → Instant Response
                ↓ (research query)
            [Clarification Agent] → Security Gate
                ↓ (proceed)
            [Router Agent] → Decides: pdf_only | web_only | hybrid
                ↓
            [Planner Agent] → Task decomposition
                ↓
            [RAG / Web / Both] → Parallel execution for hybrid
                ↓
            [Synthesis Agent] → Combines sources
                ↓
            [Reflection Agent] → Quality check (loops if score < 0.7)
                ↓
            Final Answer
```

### 8 Specialized Agents

All agents in `src/agents/` use prompts from `config.ini` (NOT hardcoded):

1. **Intent Router** - Fast-path classification (greetings/meta → instant response, research → full pipeline)
2. **Clarification** - LLM-based security + ambiguity detection (no keyword matching)
3. **Router** - Semantic routing to pdf/web/hybrid sources
4. **Planner** - Multi-step task breakdown
5. **Recursive RAG** - Follows citation chains (max depth configurable)
6. **Web Search** - Tavily wrapper with reference tracking
7. **Synthesis** - Merges PDF + web results with provenance
8. **Reflection** - Quality scoring + answer refinement

**Orchestrator** (`src/agents/orchestrator.py`): LangGraph StateGraph with conditional edges, security gates, and reflection loops

### Core Infrastructure

- **Config** (`config.ini`): All prompts, settings, RAG parameters. Use `config.py` to access
- **LLM Client** (`src/core/llm_client.py`): OpenAI GPT-4o-mini wrapper with JSON mode
- **Embeddings** (`src/core/embeddings.py`): text-embedding-ada-002 with batching
- **Vector Store** (`src/retrieval/vector_store.py`): Pinecone with auto-index creation + reranking
- **Recursive Retriever** (`src/retrieval/recursive_retriever.py`): Multi-hop citation following
- **Web Searcher** (`src/retrieval/web_searcher.py`): Tavily integration
- **Memory Management** (`src/memory/history_manager.py`): Token-based history compression with tiktoken
- **Session Manager** (`src/memory/session_manager.py`): LangGraph checkpointer for conversation state

## Critical Design Principles

### 1. Config-Driven (No Hardcoding)

**ALL agent logic comes from LLM prompts in `config.ini`**. The assignment explicitly forbids hardcoded rules like:
```python
# ❌ NEVER DO THIS
if "current" in query or "latest" in query:
    route_to_web_search()
```

Instead, agents use LLM reasoning with prompts from config:
```python
# ✅ CORRECT
routing = await router_agent.route(query)  # Uses prompt from config.ini
```

### 2. Singleton Pattern for Shared Resources

**Important**: Vector store and retrievers use lazy initialization:
```python
# Get singleton instance (initializes if needed)
vector_store = get_vector_store()
retriever = get_recursive_retriever(vector_store)
```

Never use global `vector_store` directly - always call `get_vector_store()`.

### 3. RetrievalNode vs Dict Conversion

The `RecursiveRetriever` returns `RetrievalNode` objects. Agents must convert to dicts:
```python
# In recursive_rag agent
retrieval_dicts = []
for node in retrieval_results:
    retrieval_dicts.append({
        'text': node.content,  # NOT node['text']
        'metadata': node.metadata,
        'score': node.score,
        # ...
    })
```

### 4. LangGraph State Management Pattern

**CRITICAL**: Avoid history duplication bug with `add` reducer:

```python
# ❌ NEVER DO THIS - Causes 10x history duplication
async def my_node(state: AgentState) -> AgentState:
    return {
        **state,  # This re-adds ALL fields, including history!
        "my_field": "new_value"
    }

# ✅ CORRECT - Only return fields you're updating
async def my_node(state: AgentState) -> AgentState:
    return {
        "my_field": "new_value"  # Only this field gets updated
    }
```

When `history` uses `Annotated[List, add]` reducer, spreading state causes exponential duplication.

### 5. Config.ini Syntax Rules

**CRITICAL**: ConfigParser treats `%` as interpolation marker. Must escape:

```ini
# ❌ WRONG - Causes InterpolationSyntaxError
Example: "GPT-4 achieved 92% on MMLU benchmark"

# ✅ CORRECT - Escape % as %%
Example: "GPT-4 achieved 92%% on MMLU benchmark"
```

This applies to all prompts with percentage values in examples.

### 6. Token-Based History Management

History is automatically compressed when exceeding token limits:

```python
# In orchestrator - happens automatically before each turn
compressed = await history_manager.compress_history(full_history)
# - Triggers at 10,000 tokens (configurable)
# - Keeps recent 5,000 tokens of messages
# - LLM summarizes older messages (academic-focused)
# - Summary preserves: papers, authors, metrics, user context
```

Settings in `config.ini`:
```ini
[memory]
max_history_tokens = 10000      # Trigger compression
keep_recent_tokens = 5000       # Keep in full detail
summary_target_tokens = 500     # Target summary length
```

### 7. Performance Considerations

Current optimized settings:
- **Recursion depth = 1**: Follows citations 1 level deep (reduced from 3)
- **Reflection iterations = 1**: Reflection loop runs once (reduced from 3)
- **Fast path enabled**: Intent router sends greetings/meta to instant response

Optimization settings in `config.ini`:
```ini
[rag]
max_recursion_depth = 1  # Balanced speed vs depth

[agents]
max_iterations = 1  # Single reflection pass
```

## Key Files Reference

### Configuration
- **`config.ini`** - All prompts (460+ lines), LLM/RAG/Memory settings, must escape % as %%
- **`.env`** - API keys (OPENAI_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY)
- **`src/config.py`** - Config loader with environment overrides, add dataclass for each config section

### Agent System
- **`src/agents/orchestrator.py`** - LangGraph StateGraph, conditional routing, never use `**state` spreading
- **`src/agents/intent_router.py`** - Fast-path classification for instant responses
- **`src/agents/clarification.py`** - Security validation (permissive by default)
- **`src/agents/router.py`** - Semantic routing (pdf/web/hybrid)
- **`src/agents/recursive_rag.py`** - Multi-hop retrieval
- **`src/agents/reflection.py`** - Quality control (threshold: 0.7)

### Memory System
- **`src/memory/history_manager.py`** - Token-based compression with tiktoken, LLM summarization
- **`src/memory/session_manager.py`** - LangGraph checkpointer (memory/sqlite/postgres backends)

### API & UI
- **`src/main.py`** - FastAPI app
- **`src/api/routes.py`** - Endpoints: `/ask`, `/clear_memory`, `/health`
- **`src/ui/streamlit_app.py`** - Chat playground

### Scripts
- **`scripts/ingest_pdfs.py`** - Process PDFs → Pinecone (540 chunks ingested)
- **`scripts/simple_test.py`** - End-to-end test
- **`scripts/test_history_compression.py`** - Test token-based compression with long conversations
- **`scripts/run_api.sh`** / **`scripts/run_streamlit.sh`** - Launch services

## Common Development Patterns

### Adding a New Agent

1. Create `src/agents/new_agent.py` with async methods
2. Add prompts to `config.ini` under `[prompts.new_agent]` (remember to escape % as %%)
3. Add `new_agent_system: str` and `new_agent_user: str` to `AgentPrompts` dataclass in `src/config.py`
4. Add parsing in `_parse_prompts()`: `new_agent_system=self.config.get("prompts.new_agent", "system")`
5. Access via `config.prompts.new_agent_system` and `config.prompts.new_agent_user`
6. Call LLM with `await llm_client.generate(system_prompt, user_prompt, response_format={"type": "json_object"})`
7. Export in `src/agents/__init__.py`
8. Wire into orchestrator's StateGraph (create node, add to graph, connect edges)

### Adding a New Config Section

1. Create dataclass in `src/config.py` (e.g., `MyConfig` with typed fields)
2. Add section to `config.ini`: `[my_section]` with key=value pairs
3. Create `_parse_my_config()` method returning `MyConfig` instance
4. Add `self.my_section = self._parse_my_config()` in `Config.__init__`
5. Access via `config.my_section.field_name`

### Modifying Agent Prompts

**Edit `config.ini`** only. Agents auto-load prompts on initialization. No code changes needed. Remember to escape `%` as `%%` in examples with percentages.

### Debugging Agent Flow

Check `logs/app.log` for colored LangGraph execution trace:
```
🔒 Node: Clarification + Security
✅ Query validated - proceeding
🧭 Node: Router
📍 Routing decision: hybrid (confidence: 0.85)
⚡ Execution order: parallel
📚 Node: Recursive RAG
🔍 Recursive retrieval at depth 0
...
```

### Testing Components

```python
# Test individual agent (without orchestrator)
from src.agents.router import router_agent
result = await router_agent.route("What did OpenAI release?")
print(result['decision'])  # "web_only"
```

## Environment Variables

Required in `.env`:
```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
TAVILY_API_KEY=tvly-...
```

Override any config setting with `SECTION__KEY` format:
```bash
RAG__MAX_RECURSION_DEPTH=1
AGENTS__MAX_ITERATIONS=1
```

## Common Errors & Solutions

### `'Config' object has no attribute 'X'`

**Cause**: Added new section to `config.ini` but forgot to update `config.py`

**Fix**:
1. Add `XConfig` dataclass to `config.py`
2. Create `_parse_x_config()` method
3. Add `self.x = self._parse_x_config()` in `Config.__init__`
4. If it's a prompt section, add fields to `AgentPrompts` and update `_parse_prompts()`

### `InterpolationSyntaxError: '%' must be followed by '%' or '('`

**Cause**: Unescaped `%` in `config.ini` prompts (Python ConfigParser interprets as variable)

**Fix**: Replace all `%` with `%%` in prompt examples:
```ini
# Before: "achieved 92% accuracy"
# After:  "achieved 92%% accuracy"
```

### History Growing Exponentially (10x per turn)

**Cause**: Using `**state` spreading in node return with `add` reducer

**Fix**: Only return fields you're updating:
```python
# Don't spread state
return {"field": value}  # ✅
# Never do this
return {**state, "field": value}  # ❌
```

### Semaphore Leak Warning on Shutdown

**Cause**: BGE reranker model not being cleaned up properly

**Fix**: Already implemented in `src/main.py` shutdown hook - deletes reranker and clears CUDA cache

## Known Issues & Limitations

1. **Performance**: Research queries take ~30-60s (target: <10s). Optimizations applied:
   - ✅ Reduced recursion depth to 1 (from 3)
   - ✅ Reduced reflection iterations to 1 (from 3)
   - ✅ Added intent router for fast path (greetings, meta questions)
   - Remaining: Consider caching embeddings, parallel web+RAG execution

2. **Hybrid Mode**: Sequential execution (not truly parallel) - LangGraph limitation

3. **History Compression**: First compression may take 2-3s for LLM summarization

## Future Improvements

- **Docker Setup**: Create Dockerfile + docker-compose.yml for deployment
- **README with Architecture Diagram**: Visual documentation of agent flow
- **Performance Optimization**: Target <10s per research query (currently ~30-60s)
- **Citation-Peek**: Fetch title/abstract before full PDF in recursive retrieval
- **True Parallel Execution**: Implement concurrent web+RAG for hybrid queries
- **Streaming Responses**: Stream answers token-by-token via SSE
- **Advanced Caching**: Cache embeddings and frequent queries

## Assignment Requirements Checklist

✅ Multi-agent architecture (LangGraph with 8 agents)
✅ No hardcoded logic (all prompts in config.ini)
✅ Handles ambiguous queries (clarification agent)
✅ PDF-based Q&A (recursive retrieval + citations)
✅ Web search (Tavily when out-of-scope)
✅ Session memory (LangGraph checkpointer + token-based compression)
✅ RESTful API (FastAPI: /ask, /clear_memory, /health)
✅ Intent classification (fast path for greetings/meta)
❌ Docker + docker-compose (CRITICAL - not yet implemented)
❌ README with architecture diagram (CRITICAL)
⚠️ Performance (<10s target, currently ~30-60s for research queries)

