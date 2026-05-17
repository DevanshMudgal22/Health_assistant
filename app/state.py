from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict, total=False):
    """
    Global state shared across all LangGraph nodes for HealthPilot AI.
    """

    # =====================================================
    # CHAT MEMORY (AUTO MERGED BY LANGGRAPH)
    # =====================================================
    messages: Annotated[List[BaseMessage], operator.add]

    # =====================================================
    # SESSION IDENTITY
    # =====================================================
    session_id: Optional[str]          # unique ID per chat session

    # =====================================================
    # USER PROFILE (SUPABASE AUTH)
    # =====================================================
    user_id: Optional[str]
    user_age: Optional[int]
    user_medical_history: Optional[str]

    # =====================================================
    # LONG-TERM MEMORY — ROLLING SUMMARY
    # =====================================================
    conversation_summary: Optional[str]   # LLM-generated rolling summary
    message_count: Optional[int]          # total messages so far this session
    # summarize every N messages (default 6)
    summarize_every: Optional[int]

    # =====================================================
    # CLARIFICATION (CONVERSATIONAL FLOW)
    # =====================================================
    clarification_needed: Optional[bool]   # True when bot needs more info
    clarification_questions: Optional[List[str]]  # up to 3 questions for user
    clarification_answers: Optional[dict]  # collected answers from user
    clarification_step: Optional[int]      # which question we are on (0,1,2)

    # =====================================================
    # RAG CONTEXT
    # =====================================================
    retrieved_docs: Optional[str]

    # =====================================================
    # SQL AGENT DATA
    # =====================================================
    sql_query: Optional[str]
    sql_result: Optional[str]

    # =====================================================
    # WEB SEARCH DATA
    # =====================================================
    web_search_result: Optional[str]

    # =====================================================
    # ROUTER / SAFETY
    # =====================================================
    intent: Optional[str]
    error: Optional[str]

    # =====================================================
    # OPTIONAL SYSTEM DATA
    # =====================================================
    tool_used: Optional[str]

     # =====================================================
    # INCOGNITO MODE
    # =====================================================
    incognito: Optional[bool]          # ✅ NEW — True = skip all DB saves