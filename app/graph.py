from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.nodes import (
    router_node,
    safety_node,
    clarification_node,
    summarization_node,
    rag_node,
    sql_agent_node,
    web_search_node,
    response_node,
)


def define_graph():
    """
HealthPilot AI — LangGraph Workflow

FLOW:
┌─────────┐
│  router │
└────┬────┘
     │
┌────▼──────────────────────────────────────────────────────────┐
│  greeting_done → END  (reply already in messages)             │
│  emergency     → safety → END                                 │
│  data_query    → sql_agent → response → END                   │
│  general_info  → web_search → response → END                  │
│  medical_advice → clarification                               │
│       ├─ still clarifying → END (bot asked a question)        │
│       └─ done → summarization → rag → response → END          │
└───────────────────────────────────────────────────────────────┘
"""

    workflow = StateGraph(AgentState)

    # ── Register all nodes ──
    workflow.add_node("router",        router_node)
    workflow.add_node("safety",        safety_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("summarization", summarization_node)
    workflow.add_node("rag",           rag_node)
    workflow.add_node("sql_agent",     sql_agent_node)
    workflow.add_node("web_search",    web_search_node)
    workflow.add_node("response",      response_node)

    # ── Entry point ──
    workflow.set_entry_point("router")

    # =====================================================
    # ROUTER → BRANCH DECISION
    # =====================================================

    def router_decision(state: AgentState) -> str:
        intent = (state.get("intent") or "").lower()

        if intent == "greeting_done":
            # Reply already appended to messages in router_node — stop here
            return "end"

        if intent == "emergency":
            return "safety"

        if intent == "data_query":
            return "sql_agent"

        if intent == "medical_advice":
            return "clarification"

        # general_info / unknown
        return "web_search"

    workflow.add_conditional_edges(
        "router",
        router_decision,
        {
            "end":           END,
            "safety":        "safety",
            "sql_agent":     "sql_agent",
            "clarification": "clarification",
            "web_search":    "web_search",
        },
    )

    # =====================================================
    # SAFETY → END (emergency always stops here)
    # =====================================================
    # Safety node sets error="emergency" — always terminate after safety reply
    workflow.add_edge("safety", END)

    # =====================================================
    # CLARIFICATION → STILL ASKING OR DONE
    # =====================================================

    def clarification_decision(state: AgentState) -> str:
        still_clarifying = state.get("clarification_needed", False)
        if still_clarifying:
            # Bot asked a question — wait for next user turn
            return "end"
        # All info collected — summarize conversation then run RAG
        return "summarization"

    workflow.add_conditional_edges(
        "clarification",
        clarification_decision,
        {
            "end":           END,
            "summarization": "summarization",
        },
    )

    # =====================================================
    # SUMMARIZATION → RAG → RESPONSE → END
    # =====================================================
    workflow.add_edge("summarization", "rag")
    workflow.add_edge("rag",           "response")

    # =====================================================
    # sql_agent / web_search → RESPONSE → END
    # =====================================================
    workflow.add_edge("sql_agent",  "response")
    workflow.add_edge("web_search", "response")
    workflow.add_edge("response",   END)

    return workflow.compile()