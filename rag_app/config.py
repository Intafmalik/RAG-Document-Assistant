import os
from dotenv import load_dotenv
load_dotenv()

# Base directory (rag_app/)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LiteLLM Proxy Configuration
# Overridable via env var so Docker Compose can point this at the "litellm"
# service hostname instead of localhost (see docker-compose.yml).
LITELLM_PROXY_URL = os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000")
LITELLM_MODEL_NAME = "gemini-primary"
LITELLM_FALLBACK_MODEL = "gemini-fallback"

# Default to OpenAI-compatible env var
LITELLM_API_KEY_ENV = "GEMINI_API_KEY"

# ChromaDB Configuration
CHROMA_PERSIST_DIR = os.path.join(_BASE_DIR, "..", "chroma_db")
CHROMA_COLLECTION_PREFIX = "ramayana_"

# PDF Document Configuration
PDF_PATH = os.path.join(_BASE_DIR, "rag_docs", "ramayana.pdf")
PDF_TEXT_EXTRACTOR = "pdfplumber"  # or "pypdf"

# Chunking Strategy Parameters
CHUNKING_STRATEGIES = {
    "fixed": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "separator": "\n",
    },
    "recursive": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "separators": ["\n\n", "\n", "।", " ", ""],
    },
    "semantic": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "embedding_model": "all-MiniLM-L6-v2",
    },
    "structural": {
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
}

# Retrieval Configuration
DEFAULT_RETRIEVAL_K = 10
HYBRID_K_DENSE = 10
HYBRID_K_SPARSE = 10
RRF_K = 60  # RRF constant
RERANK_TOP_K = 5  # After re-ranking, top K for LLM context
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Question Answering Configuration
MAX_CONTEXT_TOKENS = 2000
LLM_STREAMING = True
TEMPERATURE = 0.7