from functools import lru_cache

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from app.supabase_client import supabase


# =====================================================
# EMBEDDINGS (CACHED - IMPORTANT FOR PERFORMANCE)
# =====================================================

@lru_cache(maxsize=1)
def get_embeddings():
    """
    Load HuggingFace embedding model once.
    Prevents reloading model on every request.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


# =====================================================
# VECTOR STORE (CACHED INSTANCE)
# =====================================================

@lru_cache(maxsize=1)
def get_vector_store():
    """
    Initialize Supabase vector store.
    Cached to avoid repeated DB + embedding setup.
    """
    embeddings = get_embeddings()

    return SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",        # Supabase table name
        query_name="match_documents",  # RPC function in Supabase
    )


# =====================================================
# RETRIEVER
# =====================================================

def get_retriever():
    """
    Returns optimized retriever for HealthPilot AI.
    """

    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_type="mmr",  # improves diversity of results
        search_kwargs={
            "k": 4,        # number of final documents
            "fetch_k": 8,  # pool size before reranking
        },
    )