from functools import lru_cache
from supabase import create_client, Client
from typing import List, Dict, Optional
from datetime import datetime

from app.config import SUPABASE_URL, SUPABASE_KEY


# =====================================================
# SUPABASE CLIENT (CACHED SINGLETON)
# =====================================================

@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Create and return a cached Supabase client.
    lru_cache ensures only one instance is created.
    """
    url = (SUPABASE_URL or "").strip()
    key = (SUPABASE_KEY or "").strip()

    if not url or not key:
        raise ValueError(
            "[HealthPilot] SUPABASE_URL or SUPABASE_KEY is missing."
        )

    return create_client(url, key)


# Global singleton instance
supabase: Client = get_supabase_client()


# =====================================================
# CONVERSATION MEMORY HELPERS
# =====================================================

def save_message(session_id: str, role: str, content: str, intent: str = None):
    """
    Save a single message (user or assistant) to Supabase.

    Args:
        session_id : unique session string
        role       : 'user' or 'assistant'
        content    : message text
        intent     : optional router intent tag
    """
    try:
        supabase.table("conversations").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "intent": intent or "",
        }).execute()
    except Exception as e:
        print(f"[Memory] save_message error: {e}")


def load_recent_messages(session_id: str, limit: int = 10) -> List[Dict]:
    """
    Load the most recent N messages for a session.

    Returns list of dicts: [{role, content, created_at}, ...]
    """
    try:
        res = (
            supabase.table("conversations")
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        # reverse so oldest is first
        return list(reversed(res.data or []))
    except Exception as e:
        print(f"[Memory] load_recent_messages error: {e}")
        return []


def get_message_count(session_id: str) -> int:
    """
    Return total number of messages saved for a session.
    """
    try:
        res = (
            supabase.table("conversations")
            .select("id", count="exact")
            .eq("session_id", session_id)
            .execute()
        )
        return res.count or 0
    except Exception as e:
        print(f"[Memory] get_message_count error: {e}")
        return 0


# =====================================================
# SUMMARY HELPERS
# =====================================================

def save_summary(session_id: str, summary: str, message_count: int):
    """
    Upsert the rolling summary for a session.
    One summary row per session — updated each time.
    """
    try:
        if not summary:  # ✅ IMPROVEMENT
            print("[Memory] Empty summary — skipping save")
            return
        supabase.table("conversation_summaries").upsert({
            "session_id": session_id,
            "summary": summary,
            "message_count": message_count,
            "updated_at": datetime.utcnow().isoformat(),
        }, on_conflict="session_id").execute()
    except Exception as e:
        print(f"[Memory] save_summary error: {e}")


def load_summary(session_id: str) -> Optional[str]:
    """
    Load the latest rolling summary for a session.
    Returns None if no summary exists yet.
    """
    try:
        res = (
            supabase.table("conversation_summaries")
            .select("summary")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        data = res.data or []
        return data[0]["summary"] if data else None
    except Exception as e:
        print(f"[Memory] load_summary error: {e}")
        return None


# =====================================================
# CLEAR SESSION (DELETE CHAT FROM DB)
# =====================================================

def clear_session(session_id: str) -> bool:
    """
    Deletes all messages and summary for a session.
    Called when user clears chat from the app.
    """
    try:
        # Delete all messages for this session
        supabase.table("conversations") \
            .delete() \
            .eq("session_id", session_id) \
            .execute()

        # Delete the summary for this session
        supabase.table("conversation_summaries") \
            .delete() \
            .eq("session_id", session_id) \
            .execute()

        print(f"[Memory] Session cleared: {session_id}")
        return True

    except Exception as e:
        print(f"[Memory] clear_session error: {e}")
        return False