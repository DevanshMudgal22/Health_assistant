from typing import Dict, Any

from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from app.config import TAVILY_API_KEY
from app.supabase_client import supabase


# =====================================================
# TAVILY WEB SEARCH TOOL
# =====================================================

def get_search_tool() -> TavilySearch:
    """
    Returns Tavily search tool for web fallback.
    Used when RAG or database cannot answer a query.
    """

    if not TAVILY_API_KEY:
        raise ValueError("[HealthPilot Config Error] TAVILY_API_KEY missing.")

    return TavilySearch(
        tavily_api_key=TAVILY_API_KEY,
        max_results=3,
        search_depth="advanced",
    )


# =====================================================
# SAFE SUPABASE QUERY TOOL
# =====================================================

ALLOWED_TABLES = {"medicines", "doctors"}


@tool
def supabase_query_tool(params: Dict[str, Any]) -> str:
    """
    Safe database tool for HealthPilot AI.

    This tool prevents raw SQL execution and only allows
    querying approved tables.

    Example params:
    {
        "table": "medicines",
        "select": "*",
        "limit": 5
    }
    """

    try:
        table = params.get("table")
        select = params.get("select", "*")
        limit = params.get("limit", 5)

        if not table:
            return "Error: table name missing."

        if table not in ALLOWED_TABLES:
            return f"Error: table '{table}' is not allowed."

        response = (
            supabase
            .table(table)
            .select(select)
            .limit(limit)
            .execute()
        )

        data = response.data or []

        if not data:
            return "No data found."

        formatted = "\n".join(str(row) for row in data)

        return formatted

    except Exception as e:
        return f"Supabase query error: {str(e)}"


# =====================================================
# TOOL REGISTRY
# =====================================================

def get_tools():
    """
    Central registry for tools used by agents.
    """

    tools = [supabase_query_tool]

    # Add web search if API key exists
    if TAVILY_API_KEY:
        tools.append(get_search_tool())

    return tools