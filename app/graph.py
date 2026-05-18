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

UPDATED FLOW:
┌─────────┐
│  router │
└────┬────┘
     │
┌────▼──────────────────────────────────────────────┐
│  emergency → safety → END                         │
│  data_query → sql_agent → response → summarization → END      
│  general_info → web_search → response → summarization → END   
│  medical_advice → clarification_node              │
│       ├─ still clarifying → END (ask Q)           │
│       └─ done → summarization → rag               │
│                              → response → summarization → END 
└───────────────────────────────────────────────────┘
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
    def router_decision(state: AgentState):
        intent = (state.get("intent") or "").lower()

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
            "safety":        "safety",
            "sql_agent":     "sql_agent",
            "clarification": "clarification",
            "web_search":    "web_search",
        },
    )

    # =====================================================
    # SAFETY → STOP OR CONTINUE
    # =====================================================
    def safety_decision(state: AgentState):
        error = (state.get("error") or "").lower()
        if "emergency" in error:
            return END
        return "response"

    workflow.add_conditional_edges(
        "safety",
        safety_decision,
        {END: END, "response": "response"},
    )

    # =====================================================
    # CLARIFICATION → STILL ASKING OR DONE
    # =====================================================
    def clarification_decision(state: AgentState):
        """
        If clarification is still needed → stop (bot asked a question).
        If clarification is done → move to summarization then RAG.
        """
        still_clarifying = state.get("clarification_needed", False)

        if still_clarifying:
            # Bot asked a question — wait for user reply
            return END

        # All answers collected — proceed to summarize + RAG
        return "summarization"

    workflow.add_conditional_edges(
        "clarification",
        clarification_decision,
        {END: END, "summarization": "summarization"},
    )

    # =====================================================
    # SUMMARIZATION → RAG
    # =====================================================
    workflow.add_edge("summarization", "rag")

    # =====================================================
    # ALL PATHS → RESPONSE → END
    # =====================================================
    workflow.add_edge("sql_agent",  "response")
    workflow.add_edge("web_search", "response")
    workflow.add_edge("rag",        "response")


    return workflow.compile()