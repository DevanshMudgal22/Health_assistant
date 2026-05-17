from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def _get_env(key: str, required: bool = True) -> str | None:
    """
    Safely fetch environment variables.

    Args:
        key: Environment variable name
        required: If True, raises error when missing

    Returns:
        str | None
    """
    value = os.getenv(key)

    if required and (value is None or value.strip() == ""):
        raise ValueError(f"[HealthPilot Config Error] Missing required env variable: {key}")

    return value


# ===== REQUIRED KEYS =====
GROQ_API_KEY: str = _get_env("GROQ_API_KEY")
SUPABASE_URL: str = _get_env("SUPABASE_URL")
SUPABASE_KEY: str = _get_env("SUPABASE_KEY")

# ===== OPTIONAL KEYS =====
# Web search fallback may be optional during local dev
TAVILY_API_KEY: str | None = _get_env("TAVILY_API_KEY", required=False)
