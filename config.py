
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Load local environment variables
load_dotenv(PROJECT_ROOT / ".env", override=True)


def get_setting(name: str, default=""):
    """
    Read configuration from environment variables or Streamlit secrets.

    Supports:
    1. Local development using .env
    2. Streamlit Community Cloud using st.secrets
    """

    value = os.getenv(name)

    if value is not None and value.strip() != "":
        return value

    try:
        import streamlit as st
        value = st.secrets.get(name, default)

        if value is not None and str(value).strip() != "":
            return value

    except Exception:
        pass

    return default


def _get_int(name: str, default: int) -> int:
    """Read an integer setting with a default fallback."""

    raw_value = get_setting(name, default)

    try:
        return int(raw_value)
    except (ValueError, TypeError):
        print(
            f"[config] '{name}={raw_value}' is not a number. "
            f"Using default -> {default}"
        )
        return default


# Project directories
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
FAISS_DIR = VECTORSTORE_DIR / "faiss"
DOC_STORE_PATH = VECTORSTORE_DIR / "doc_store.json"


# OpenAI configuration
GROQ_API_KEY = get_setting("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = get_setting(
    "GROQ_CHAT_MODEL", "gpt-4o-mini"
)
HF_EMBEDDING_MODEL = get_setting(
    "HF_EMBEDDING_MODEL", "text-embedding-3-small"
)

LLM_TEMPERATURE = 0.0


# Vector database configuration
VECTOR_DB = get_setting("VECTOR_DB", "faiss").strip().lower()

PINECONE_API_KEY = get_setting("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = get_setting(
    "PINECONE_INDEX_NAME", "advanced-rag"
)
PINECONE_NAMESPACE = get_setting(
    "PINECONE_NAMESPACE", "default"
)


# Chunking configuration
PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE", 1600)
PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 200)

CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", 400)
CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)


# Retrieval configuration
NUM_QUERY_VARIANTS = _get_int("NUM_QUERY_VARIANTS", 3)
VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 6)
BM25_TOP_K = _get_int("BM25_TOP_K", 6)
HYBRID_TOP_K = _get_int("HYBRID_TOP_K", 10)
RERANK_TOP_K = _get_int("RERANK_TOP_K", 4)

MIN_RERANK_SCORE = float(get_setting("MIN_RERANK_SCORE", "-4.0"))
MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 8000)


# Reranker configuration
RERANKER = get_setting(
    "RERANKER", "cross-encoder"
).strip().lower()

CROSS_ENCODER_MODEL = get_setting(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# Conversation memory
MEMORY_WINDOW = _get_int("MEMORY_WINDOW", 4)


# Validation functions
def check_openai_key() -> None:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not found."
        )


def check_pinecone_config() -> None:
    if not PINECONE_API_KEY:
        raise ValueError(
            "VECTOR_DB=pinecone but PINECONE_API_KEY is empty."
        )

    if not PINECONE_INDEX_NAME:
        raise ValueError(
            "VECTOR_DB=pinecone but PINECONE_INDEX_NAME is empty."
        )


# Application settings summary
def describe_settings() -> dict:
    return {
        "Chat Model": GROQ_CHAT_MODEL,
        "Embedding Model": HF_EMBEDDING_MODEL,
        "VectorDB": VECTOR_DB,
        "Parent Chunk Size": PARENT_CHUNK_SIZE,
        "Child Chunk Size": CHILD_CHUNK_SIZE,
    }
