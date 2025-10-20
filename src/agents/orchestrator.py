"""LangGraph Orchestrator - Coordinates all agents with parallel execution"""

import asyncio
from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, END
from loguru import logger

from src.config import config
from src.memory.session_manager import session_manager
from src.memory.history_manager import history_manager
from src.agents.intent_router import intent_router_agent
from src.agents.clarification import clarification_agent
from src.agents.router import router_agent
from src.agents.planner import planner_agent
from src.agents.recursive_rag import recursive_rag_agent
from src.agents.web_search import web_search_agent
from src.agents.synthesis import synthesis_agent
from src.agents.reflection import reflection_agent


# ============================================
# STATE DEFINITION
# ============================================

class AgentState(TypedDict):
    """State shared across all agents"""
    # Input
    query: str
    history: Annotated[List[Dict[str, Any]], add]  # Use add reducer to accumulate messages
    history_summary: Optional[str]  # Summary of old conversation history
    session_id: str

    # Intent Router
    intent: Optional[str]  # "greeting" | "meta_question" | "simple_info" | "research_query"
    intent_confidence: float
    fast_response: Optional[str]  # Pre-generated response for fast-path intents

    # Clarification
    validation_result: Optional[Dict[str, Any]]
    is_safe: bool
    needs_clarification: bool
    clarification_questions: Optional[List[str]]

    # Routing
    routing_decision: Optional[str]  # "pdf_only" | "web_only" | "hybrid"
    execution_order: Optional[str]  # "parallel" | "sequential"
    routing_confidence: float

    # Planning
    plan: Optional[Dict[str, Any]]
    is_complex: bool

    # Retrieval results
    rag_results: Optional[Dict[str, Any]]
    web_results: Optional[Dict[str, Any]]

    # Synthesis
    answer: Optional[str]
    sources: Optional[List[Dict[str, Any]]]

    # Reflection
    quality_score: float
    reflection_result: Optional[Dict[str, Any]]
    iteration: int

    # Control
    should_continue: bool
    error: Optional[str]


# ============================================
# AGENT NODE FUNCTIONS
# ============================================

async def intent_router_node(state: AgentState) -> AgentState:
    """Classify query intent for fast-path routing"""
    logger.info("🎯 Node: Intent Router")

    intent_result = await intent_router_agent.classify_intent(
        query=state['query'],
        history=state.get('history', []),
        history_summary=state.get('history_summary', '')
    )

    # Only return fields we're updating (don't spread state to avoid history duplication)
    return {
        "intent": intent_result.get('intent', 'research_query'),
        "intent_confidence": intent_result.get('confidence', 0.0),
        "fast_response": intent_result.get('suggested_response')
    }


async def fast_response_node(state: AgentState) -> AgentState:
    """Return pre-generated fast response"""
    logger.info(f"⚡ Node: Fast Response ({state.get('intent')})")

    return {
        "answer": state.get('fast_response', config.system.default_greeting),
        "sources": [],
        "quality_score": 1.0,
        "iteration": 0
    }


async def clarification_node(state: AgentState) -> AgentState:
    """Validate query for security and clarity"""
    logger.info("🔒 Node: Clarification + Security")

    validation = await clarification_agent.validate(
        query=state['query'],
        history=state.get('history', []),
        history_summary=state.get('history_summary', '')
    )

    return {
        "validation_result": validation,
        "is_safe": validation.get('is_safe', True),
        "needs_clarification": validation.get('needs_clarification', False),
        "clarification_questions": validation.get('clarification_questions', [])
    }


async def router_node(state: AgentState) -> AgentState:
    """Route query to appropriate information sources"""
    logger.info("🧭 Node: Router")

    routing = await router_agent.route(
        query=state['query'],
        history=state.get('history', []),
        context="",
        history_summary=state.get('history_summary', '')
    )

    return {
        "routing_decision": routing.get('decision'),
        "execution_order": routing.get('execution_order', 'parallel'),
        "routing_confidence": routing.get('confidence', 0.0)
    }


async def planner_node(state: AgentState) -> AgentState:
    """Create execution plan"""
    logger.info("📋 Node: Planner")

    plan = await planner_agent.plan(
        query=state['query'],
        history=state.get('history', []),
        history_summary=state.get('history_summary', '')
    )

    return {
        "plan": plan,
        "is_complex": plan.get('is_complex', False)
    }


async def rag_node(state: AgentState) -> AgentState:
    """Recursive RAG retrieval"""
    logger.info("📚 Node: Recursive RAG")

    rag_results = await recursive_rag_agent.retrieve(
        query=state['query'],
        history=state.get('history', []),
        depth=0,
        parent_context="",
        history_summary=state.get('history_summary', '')
    )

    return {
        "rag_results": rag_results
    }


async def web_search_node(state: AgentState) -> AgentState:
    """Web search"""
    logger.info("🌐 Node: Web Search")

    web_results = await web_search_agent.search(
        query=state['query']
    )

    return {
        "web_results": web_results
    }


async def hybrid_parallel_node(state: AgentState) -> AgentState:
    """Run RAG and Web Search in parallel for hybrid queries"""
    logger.info("⚡ Node: Hybrid Parallel Execution")

    # Run both retrieval methods concurrently
    rag_task = recursive_rag_agent.retrieve(
        query=state['query'],
        history=state.get('history', []),
        depth=0,
        parent_context="",
        history_summary=state.get('history_summary', '')
    )

    web_task = web_search_agent.search(
        query=state['query']
    )

    # Wait for both to complete
    rag_results, web_results = await asyncio.gather(rag_task, web_task)

    logger.success("✅ Parallel execution completed")

    return {
        "rag_results": rag_results,
        "web_results": web_results
    }


async def synthesis_node(state: AgentState) -> AgentState:
    """Synthesize results from all sources"""
    logger.info("🔄 Node: Synthesis")

    synthesis = await synthesis_agent.synthesize(
        query=state['query'],
        rag_results=state.get('rag_results'),
        web_results=state.get('web_results'),
        history=state.get('history', []),
        history_summary=state.get('history_summary', '')
    )

    return {
        "answer": synthesis.get('answer'),
        "sources": synthesis.get('sources', [])
    }


async def reflection_node(state: AgentState) -> AgentState:
    """Reflect on answer quality"""
    logger.info("🪞 Node: Reflection")

    reflection = await reflection_agent.reflect(
        query=state['query'],
        answer=state.get('answer', ''),
        sources=state.get('sources', [])
    )

    # Use revised answer if provided
    final_answer = state.get('answer', '')
    if reflection.get('revised_answer'):
        final_answer = reflection['revised_answer']
        logger.info("✏️ Answer revised by reflection agent")

    return {
        "answer": final_answer,
        "quality_score": reflection.get('quality_score', 0.0),
        "reflection_result": reflection,
        "iteration": state.get('iteration', 0) + 1
    }


# ============================================
# CONDITIONAL ROUTING FUNCTIONS
# ============================================

def route_by_intent(state: AgentState) -> Literal["fast_response", "full_pipeline"]:
    """Route based on intent classification"""
    intent = state.get('intent', 'research_query')

    if intent in ['greeting', 'meta_question']:
        logger.info(f"⚡ Fast path: {intent}")
        return "fast_response"

    logger.info(f"🔄 Full pipeline: {intent}")
    return "full_pipeline"


def should_block(state: AgentState) -> Literal["block", "clarify", "proceed"]:
    """Security gate: Check if query should be blocked"""
    if not state.get('is_safe', True):
        logger.warning("🚨 Query blocked by security gate")
        return "block"

    if state.get('needs_clarification', False):
        logger.info("❓ Query needs clarification")
        return "clarify"

    return "proceed"


def route_by_decision(state: AgentState) -> Literal["pdf_only", "web_only", "hybrid"]:
    """Route based on router decision"""
    decision = state.get('routing_decision', 'pdf_only')
    logger.info(f"📍 Routing to: {decision}")
    return decision


def should_reflect_again(state: AgentState) -> Literal["reflect_again", "finish"]:
    """Check if reflection loop should continue"""
    quality_score = state.get('quality_score', 0.0)
    iteration = state.get('iteration', 0)
    max_iterations = config.agents.max_iterations
    quality_threshold = config.agents.orchestrator_quality_threshold

    if quality_score >= quality_threshold:
        logger.success(f"✅ Quality threshold met (score: {quality_score:.2f})")
        return "finish"

    if iteration >= max_iterations:
        logger.warning(f"⚠️ Max iterations ({max_iterations}) reached")
        return "finish"

    logger.info(f"🔄 Quality score {quality_score:.2f} < {quality_threshold}, reflecting again (iteration {iteration}/{max_iterations})")
    return "reflect_again"


# ============================================
# GRAPH CONSTRUCTION
# ============================================

def create_agent_graph():
    """Create the LangGraph StateGraph"""

    # Initialize graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("fast_response", fast_response_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("hybrid_parallel", hybrid_parallel_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("reflection", reflection_node)

    # Set entry point - Intent Router is first
    workflow.set_entry_point("intent_router")

    # Intent Router → Fast path or full pipeline
    workflow.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "fast_response": "fast_response",  # Greetings/meta → instant response
            "full_pipeline": "clarification"  # Research queries → full pipeline
        }
    )

    # Fast response → END
    workflow.add_edge("fast_response", END)

    # Clarification → Security gate
    workflow.add_conditional_edges(
        "clarification",
        should_block,
        {
            "block": END,  # Blocked queries end here
            "clarify": END,  # Return clarification questions
            "proceed": "router"  # Safe queries continue
        }
    )

    # Router → Planner
    workflow.add_edge("router", "planner")

    # Planner → Route by decision
    workflow.add_conditional_edges(
        "planner",
        route_by_decision,
        {
            "pdf_only": "rag",
            "web_only": "web_search",
            "hybrid": "hybrid_parallel"  # TRUE PARALLEL EXECUTION
        }
    )

    # All paths converge at synthesis
    workflow.add_edge("rag", "synthesis")
    workflow.add_edge("web_search", "synthesis")
    workflow.add_edge("hybrid_parallel", "synthesis")

    # Synthesis → Reflection
    workflow.add_edge("synthesis", "reflection")

    # Reflection → Quality gate
    workflow.add_conditional_edges(
        "reflection",
        should_reflect_again,
        {
            "reflect_again": "synthesis",  # Loop back to synthesis
            "finish": END
        }
    )

    return workflow


# ============================================
# ORCHESTRATOR CLASS
# ============================================

class Orchestrator:
    """Main orchestrator for the multi-agent system"""

    def __init__(self):
        self.graph = create_agent_graph()
        # Use shared checkpointer from session_manager for proper state persistence
        self.checkpointer = session_manager.get_checkpointer()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
        logger.info(f"🎭 Orchestrator initialized with shared checkpointer (backend={session_manager.backend})")

    async def ask(
        self,
        query: str,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process a query through the multi-agent system.

        Args:
            query: User query
            session_id: Session ID for conversation history (thread_id for checkpointer)

        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"🎬 Processing query: {query[:100]}...")

        # Run the graph with thread_id for session persistence
        config_dict = {"configurable": {"thread_id": session_id}}

        try:
            # Get previous state to access full history
            prev_state = await self.app.aget_state(config_dict)
            full_history = []
            if prev_state and prev_state.values:
                full_history = prev_state.values.get('history', [])
                logger.info(f"📜 Previous history length: {len(full_history)}")

            # Compress history if needed (token-based)
            compressed = await history_manager.compress_history(full_history)

            # New user message for this turn
            conversation_turn = [{
                "role": "user",
                "content": query
            }]

            # Initial state with compressed history and summary
            # Note: history field uses `add` reducer, so conversation_turn will append
            initial_state: AgentState = {
                "query": query,
                "history": conversation_turn,  # Will be ADDED to existing history by reducer
                "history_summary": compressed.get('summary', ''),  # Summary of old messages
                "session_id": session_id,
            }

            # Log compression stats
            if compressed.get('compressed'):
                stats = compressed.get('compression_stats', {})
                logger.info(
                    f"🗜️ History compressed: {stats.get('original_messages')} msgs → "
                    f"{stats.get('kept_messages')} recent msgs + summary "
                    f"(compression: {stats.get('compression_ratio')})"
                )

            final_state = await self.app.ainvoke(initial_state, config_dict)

            # Debug: Check history after invoke
            logger.info(f"📜 Final history length: {len(final_state.get('history', []))}")
            if final_state.get('history'):
                logger.info(f"📜 Final history sample: {final_state['history'][-2:]}")  # Show last 2 entries

            # Add assistant response to history for next turn
            # This will be automatically persisted by the checkpointer
            if final_state.get('answer'):
                response_turn = [{
                    "role": "assistant",
                    "content": final_state.get('answer', '')
                }]

                # Update state with assistant response (will be saved by checkpointer)
                await self.app.aupdate_state(
                    config_dict,
                    {"history": response_turn}  # Will be ADDED to history by reducer
                )

            # Check if blocked
            if not final_state.get('is_safe', True):
                logger.warning("🚨 Query was blocked")
                return {
                    "status": "blocked",
                    "reason": final_state.get('validation_result', {}).get('security_reasoning', 'Security threat detected'),
                    "answer": None,
                    "sources": []
                }

            # Check if clarification needed
            if final_state.get('needs_clarification', False):
                logger.info("❓ Clarification needed")
                return {
                    "status": "clarification_needed",
                    "questions": final_state.get('clarification_questions', []),
                    "issues": final_state.get('validation_result', {}).get('clarity_issues', []),
                    "answer": None,
                    "sources": []
                }

            # Success
            logger.success("✅ Query processed successfully")
            return {
                "status": "success",
                "answer": final_state.get('answer', 'No answer generated'),
                "sources": final_state.get('sources', []),
                "quality_score": final_state.get('quality_score', 0.0),
                "routing_decision": final_state.get('routing_decision'),
                "iterations": final_state.get('iteration', 0),
                "metadata": {
                    "is_complex": final_state.get('is_complex', False),
                    "routing_confidence": final_state.get('routing_confidence', 0.0),
                    "num_sources": len(final_state.get('sources', []))
                }
            }

        except Exception as e:
            logger.error(f"❌ Error in orchestrator: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "answer": None,
                "sources": []
            }

    def clear_memory(self, session_id: str):
        """Clear conversation history for a session"""
        # InMemorySaver doesn't have explicit clear method
        # To clear: user should use a new session_id
        logger.info(f"🗑️ Clear memory requested for session: {session_id} (use new session_id for fresh start)")


# Singleton instance
orchestrator = Orchestrator()
