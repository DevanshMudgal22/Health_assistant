import uuid
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage
from app.graph import define_graph
from app.supabase_client import (
    load_recent_messages,
    load_summary,
    get_message_count,
    clear_session,
)

app = FastAPI(title="HealthPilot AI")

graph = define_graph()


# =====================================================
# REQUEST / RESPONSE MODELS
# =====================================================

class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    clarification_step: int = 0
    clarification_questions: list = []
    clarification_answers: dict = {}
    incognito: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str
    is_emergency: bool = False          # deduplicated (was declared twice before)
    clarification_needed: bool = False
    clarification_step: int = 0
    clarification_questions: list = []
    clarification_answers: dict = {}


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
def home():
    return {"status": "HealthPilot AI is running ✅"}


# =====================================================
# CHAT ENDPOINT
# =====================================================

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint.

    - Creates a new session_id if not provided
    - Loads rolling summary from Supabase for long-term memory
    - Loads recent messages to rebuild conversation context
    - Runs LangGraph workflow
    - Returns response + session state for frontend to track
    """

    # ── Session Management ──
    session_id = req.session_id or str(uuid.uuid4())

    # ── Load Long-term Memory from Supabase ──
    existing_summary = load_summary(session_id) or ""
    message_count    = get_message_count(session_id)

    # ── Rebuild recent message history ──
    recent_msgs = load_recent_messages(session_id, limit=10)

    history = []
    for m in recent_msgs:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    # Add current user message
    history.append(HumanMessage(content=req.message))

    # ── Build Initial State ──
    state = {
        "messages":               history,
        "session_id":             session_id,
        "conversation_summary":   existing_summary,
        "message_count":          message_count,
        "summarize_every":        6,

        # Pass clarification state from frontend
        "clarification_step":      req.clarification_step,
        "clarification_questions": req.clarification_questions,
        "clarification_answers":   req.clarification_answers,
        "clarification_needed":    False,
        "incognito":               req.incognito,
    }

    # ── Run LangGraph ──
    result = graph.invoke(state)

    # ── Extract AI Response ──
    messages    = result.get("messages", [])
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    response_text = (
        ai_messages[-1].content
        if ai_messages
        else "I'm sorry, I couldn't generate a response."
    )

    # ── Detect emergency ──
    is_emergency = (
        result.get("intent") == "emergency"
        or "emergency" in (result.get("error") or "").lower()
    )

    # ── Return response + clarification state for frontend ──
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        is_emergency=is_emergency,
        clarification_needed=result.get("clarification_needed", False),
        clarification_step=result.get("clarification_step", 0),
        clarification_questions=result.get("clarification_questions", []),
        clarification_answers=result.get("clarification_answers", {}),
    )


# =====================================================
# SESSION HISTORY ENDPOINT
# =====================================================

@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Returns recent messages for a session."""
    messages = load_recent_messages(session_id, limit=50)
    summary  = load_summary(session_id)

    return {
        "session_id": session_id,
        "summary": summary,
        "messages": messages,
    }


# =====================================================
# CLEAR CHAT ENDPOINT
# =====================================================

@app.delete("/clear/{session_id}")
def clear_chat(session_id: str):
    """Deletes all messages and summary for a session from Supabase."""
    success = clear_session(session_id)
    if success:
        return {"status": "cleared", "session_id": session_id}
    return {"status": "error", "message": "Failed to clear chat"}