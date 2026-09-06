"""
FastAPI backend for the RAG assistant (Task 2 - Productionization).

Wraps the existing RAGPipeline (rag_app/) with:
- Async request handling
- In-memory response caching (question+strategy -> answer), TTL-based
- Retry with exponential backoff around the LLM call (tenacity)
- Per-IP rate limiting (slowapi)
- Structured error handling / graceful degradation (never 500s on LLM failure -
  returns a degraded RAGAnswer instead, same contract as rag_app.qa_pipeline)

Run directly for local dev:
    uv run uvicorn service.api:app --reload --port 8000

Or via Docker (see docker-compose.yml).
"""
import time
import hashlib
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from rag_app.vector_store import VectorStoreManager
from rag_app.retrieval.hybrid import HybridRetriever, CrossEncoderReranker
from rag_app.qa_pipeline import RAGPipeline, RAGAnswer
from rag_app import config

# --------------------------------------------------------------------------
# Rate limiter
# --------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# --------------------------------------------------------------------------
# Simple in-memory TTL cache (question+strategy+model -> RAGAnswer)
# For multi-instance deployments, swap this for Redis (see README "Scaling"
# section) - kept in-memory here to avoid adding infra for a course project.
# --------------------------------------------------------------------------
class TTLCache:
    def __init__(self, ttl_seconds: int = 300, max_entries: int = 256):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[str, tuple[float, RAGAnswer]] = {}

    @staticmethod
    def _key(question: str, strategy: str, model: str) -> str:
        raw = f"{question.strip().lower()}|{strategy}|{model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, question: str, strategy: str, model: str) -> Optional[RAGAnswer]:
        key = self._key(question, strategy, model)
        entry = self._store.get(key)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        return value

    def set(self, question: str, strategy: str, model: str, value: RAGAnswer) -> None:
        if len(self._store) >= self.max_entries:
            # Evict the oldest entry (simple FIFO eviction)
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]
        key = self._key(question, strategy, model)
        self._store[key] = (time.time(), value)


cache = TTLCache(ttl_seconds=300)

# --------------------------------------------------------------------------
# Pipeline components - built once at startup, reused across requests.
# Building the embedder/vector store per-request would be extremely slow.
# --------------------------------------------------------------------------
_pipelines: dict[str, RAGPipeline] = {}
_embedder = None
_vs_manager = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(
            config.CHUNKING_STRATEGIES["semantic"]["embedding_model"]
        )
    return _embedder


def _get_vs_manager():
    global _vs_manager
    if _vs_manager is None:
        _vs_manager = VectorStoreManager(config.CHROMA_PERSIST_DIR)
    return _vs_manager


def _build_bm25_corpus() -> list[str]:
    """Mirror the corpus construction used by the CLI's `ask` command."""
    import re
    from rag_app.document_loader import RamayanaPDFLoader

    loader = RamayanaPDFLoader(config.PDF_PATH)
    full_text = loader.load_all_text()
    sentences = re.split(r"[.!?]+", full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunk_size = 200
    corpus = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i:min(i + chunk_size, len(sentences))]) + "."
        corpus.append(chunk)
    return corpus


def get_pipeline(strategy: str, model: str, enable_tools: bool) -> RAGPipeline:
    """Get or lazily build a RAGPipeline for a given (strategy, model) pair."""
    cache_key = f"{strategy}:{model}:{enable_tools}"
    if cache_key in _pipelines:
        return _pipelines[cache_key]

    vs_manager = _get_vs_manager()
    collection = vs_manager.get_collection(strategy)
    if collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No documents indexed for strategy '{strategy}'. "
                f"Run `python -m rag_app build --strategy {strategy}` first."
            ),
        )

    embedder = _get_embedder()
    bm25_corpus = _build_bm25_corpus()

    hybrid_retriever = HybridRetriever(
        vector_store=vs_manager,
        bm25_corpus=bm25_corpus,
        embedding_function=embedder.encode,
        k_dense=config.DEFAULT_RETRIEVAL_K,
        k_sparse=config.DEFAULT_RETRIEVAL_K,
        rrf_k=config.RRF_K,
    )
    reranker = CrossEncoderReranker()

    pipeline = RAGPipeline(
        vector_store=vs_manager,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm_model=model,
        strategy=strategy,
        enable_tools=enable_tools,
    )
    _pipelines[cache_key] = pipeline
    return pipeline


# --------------------------------------------------------------------------
# Retry wrapper around the (blocking) pipeline.ask call.
# tenacity retries on generic Exception here because RAGPipeline.ask()
# already catches LLM-level errors internally and returns a degraded
# RAGAnswer rather than raising - so an exception escaping ask() means a
# genuine infrastructure problem (e.g. transient network blip) worth retrying.
# --------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _ask_with_retry(pipeline: RAGPipeline, question: str, top_k: int) -> RAGAnswer:
    return pipeline.ask(question, top_k=top_k)


async def ask_async(pipeline: RAGPipeline, question: str, top_k: int) -> RAGAnswer:
    """Run the blocking pipeline call in a worker thread so the event loop
    stays free to serve other concurrent requests (async/concurrency
    requirement)."""
    return await asyncio.to_thread(_ask_with_retry, pipeline, question, top_k)


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the embedder once at startup rather than on first request,
    # so the first user-facing request isn't slowed down by a cold model load.
    try:
        _get_embedder()
    except Exception as e:
        print(f"[WARNING] Embedder warmup failed at startup: {e}")
    yield


app = FastAPI(
    title="RAG Assistant API",
    description="Production backend for the document-QA RAG assistant",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your UI's origin in real deployments
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    strategy: str = "recursive"
    model: str = "gemini-primary"
    top_k: int = 5
    enable_tools: bool = False


class AskResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[str]
    reasoning: Optional[str] = None
    cached: bool = False


@app.get("/health")
async def health():
    """Liveness/readiness probe."""
    return {"status": "ok"}


@app.get("/stats")
@limiter.limit("30/minute")
async def stats(request: Request):
    """Vector store statistics across all built strategies."""
    vs_manager = _get_vs_manager()
    return vs_manager.get_collection_stats()


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
async def ask(request: Request, body: AskRequest):
    """Answer a question using the RAG pipeline.

    Reliability behavior:
    - Rate limited to 10 requests/minute per client IP
    - Cached responses served instantly for repeated questions (5 min TTL)
    - Retries transient failures up to 3x with exponential backoff
    - On persistent LLM failure, returns HTTP 200 with a degraded RAGAnswer
      (confidence=0.0, reasoning explains the failure) rather than a 500 -
      the UI can always render *something* instead of crashing.
    """
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=422, detail="`question` must not be empty.")

    cached = cache.get(body.question, body.strategy, body.model)
    if cached is not None:
        return AskResponse(**cached.model_dump(), cached=True)

    try:
        pipeline = get_pipeline(body.strategy, body.model, body.enable_tools)
        rag_answer = await ask_async(pipeline, body.question, body.top_k)
    except HTTPException:
        raise
    except Exception as e:
        # Graceful degradation: never let an unhandled exception 500 the
        # request - the frontend should always get a well-formed answer shape.
        return AskResponse(
            answer="The assistant is temporarily unavailable. Please try again shortly.",
            confidence=0.0,
            sources=[],
            reasoning=f"Unhandled error after retries: {e}",
            cached=False,
        )

    cache.set(body.question, body.strategy, body.model, rag_answer)
    return AskResponse(**rag_answer.model_dump(), cached=False)
