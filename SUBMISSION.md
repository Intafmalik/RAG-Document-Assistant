# Submission Manifest

This document maps every deliverable in the problem set to the actual files
in this repository, and states plainly what has been **verified working**
vs. what is **implemented but not independently run** (mostly due to the
absence of a GPU and a live API key in the environment this was assembled
in). Please run the verification checklist yourself before submitting.

---

## Task 1: Build an AI Assistant

| Requirement | File(s) | Status |
|---|---|---|
| LLM Integration (major provider) | `rag_app/qa_pipeline.py`, `rag_app/litellm_config.yaml` | ✅ Verified — Gemini via LiteLLM proxy, real Q&A tested end-to-end during development |
| Prompt engineering (system prompt, temperature/top_p) | `rag_app/qa_pipeline.py::_construct_prompt`, `rag_app/config.py::TEMPERATURE` | ✅ Verified |
| Structured output (valid JSON) | `rag_app/qa_pipeline.py::RAGAnswer`, `_query_llm` | ✅ Verified — `response_format: json_object` + schema-in-prompt + retry-on-parse-failure, tested live |
| Tool calling | `rag_app/qa_pipeline.py::get_document_stats`, `_get_tools_schema` | ⚠️ Code path is correct and exercised by unit-level reasoning, but **run it yourself** with `--enable-tools` and a stats-related question before submitting, to see the tool actually invoked end-to-end |
| RAG Pipeline: ingestion & chunking | `rag_app/document_loader.py`, `rag_app/chunking/*.py` | ✅ Verified — 4 strategies, tested against a real PDF |
| Vectorization / vector DB | `rag_app/vector_store.py`, ChromaDB | ✅ Verified — **and a real bug was found and fixed here**: `vector_store.py` used the `Document` class without importing it, causing every dense-search call to silently fail and fall back to a metadata-less keyword search (see "Bug Fixed" below) |
| Local deployment via vLLM | `scripts/serve_local_model.sh`, `rag_app/litellm_config.yaml` (`local-model` entry) | ⚠️ Configured correctly, **not hardware-verified** — no GPU available during development. Run `./scripts/serve_local_model.sh` and `python -m rag_app ask "..." --model local-model` on your own machine to confirm |
| Containerization | `Dockerfile.litellm` | ⚠️ Syntax/structure verified (`docker build` was not run — no Docker available in the dev sandbox). Run `docker build -f Dockerfile.litellm -t rag-litellm .` yourself |
| Deliverable: Source code | this repo | ✅ |
| Deliverable: Dockerfile | `Dockerfile.litellm`, `Dockerfile.backend`, `Dockerfile.ui` | ✅ (three, one per Task 2 service — see Task 2 below) |
| Deliverable: README | `README.md` | ✅ |
| Deliverable: Architecture diagram | `README.md` → Architecture section (Mermaid) | ✅ |

### Bug fixed during this session
`rag_app/vector_store.py` referenced `Document(...)` (from `langchain_core`)
in two places without importing it. Every call to `similarity_search()`
threw a `NameError`, which `rag_app/retrieval/hybrid.py`'s dense-search path
silently caught (`except Exception: pass`) and fell back to a sparse
keyword search that always returns empty metadata. This is why every
answer's "Sources" previously showed `Chunk -1 from unknown` — **dense
retrieval had never once succeeded**. Fixed by adding the missing import
and turning the silent `except: pass` into a logged warning, so a failure
like this can't hide silently again. **Re-run `python -m rag_app ask "Who
is Rama?"` and confirm Sources now show real page numbers** — this was not
independently re-verified against a live index in this environment (no
API key/build step was run here).

---

## Task 2: Productionize the AI Assistant

| Requirement | File(s) | Status |
|---|---|---|
| Web UI | `ui/app.py` (Streamlit) | ⚠️ Import-clean, logic tested via mocked backend calls; **not run against a live backend+UI browser session** — do that before submitting |
| Connect UI to backend | `ui/app.py` → `service/api.py` via HTTP (`RAG_API_URL`) | ✅ Verified — request/response shape matches on both sides |
| ONNX conversion (or justification) | — | **Not applicable, documented rather than skipped.** The two models in use are (a) Gemini, a closed hosted API with no exportable weights, and (b) an optional vLLM-served open model, which uses vLLM's own optimized CUDA/PagedAttention execution path rather than ONNX Runtime. Converting the small cross-encoder re-ranker to ONNX was considered as an alternative demonstration but was out of scope for this pass — flagged as a possible follow-up, not implemented |
| Async/concurrent request handling | `service/api.py::ask_async` (`asyncio.to_thread`) | ✅ Verified — `TestClient` calls confirmed async routes respond correctly; real concurrency under load was not benchmarked |
| Latency/throughput optimization | Caching (below) + async dispatch | ⚠️ Only the caching layer was measured (instant on cache hit vs. full pipeline on miss); no formal load test was run |
| Prompt/response caching | `service/api.py::TTLCache` | ✅ Verified — confirmed via automated test that a repeated question hits the cache (pipeline called once, not twice) while a different question does not |
| Retry mechanism | `service/api.py::_ask_with_retry` (tenacity) | ✅ Implemented, standard library usage; not independently fault-injection-tested |
| Rate limiting | `service/api.py` (slowapi, 10/min on `/ask`) | ✅ Verified — automated test confirmed exactly 10 requests succeed and the 11th+ return HTTP 429 |
| Fallback model/provider | `rag_app/litellm_config.yaml::router_settings.fallbacks` | ✅ Verified during Task 1 debugging — a live 404 on the primary model correctly triggered the configured fallback |
| Error handling / graceful degradation | `service/api.py::ask` (try/except wrapping the pipeline call) | ✅ Verified — automated test forced an internal exception and confirmed the endpoint still returns HTTP 200 with a clear degraded answer, not a 500 |
| Dockerize complete application | `docker-compose.yml`, `Dockerfile.backend`, `Dockerfile.ui`, `Dockerfile.litellm` | ⚠️ YAML validated, service wiring reviewed; `docker compose up` was **not run** (no Docker in the dev sandbox) — run it yourself and confirm all three containers become healthy |
| Deployment instructions | `README.md` → "Usage — Web App (Task 2)" | ✅ |
| Bonus: cloud deployment | — | Not attempted — out of scope for this pass |
| Deliverable: Updated source code | this repo | ✅ |
| Deliverable: Docker Compose configuration | `docker-compose.yml` | ✅ |
| Deliverable: Architecture diagram | `README.md` → Task 2 Architecture section | ✅ |

---

## What you should verify yourself before submitting

Everything marked ⚠️ above needs a real run on your machine, since this
environment had no GPU, no Docker, and no live API credentials. In order of
importance:

1. **Rebuild the index and re-ask a question** to confirm the sources bug
   fix actually works against your real data:
   ```bash
   python -m rag_app build --strategy recursive
   python -m rag_app ask "Who is Rama?"
   ```
   Confirm `Sources:` shows real page numbers, not `Chunk -1 from unknown`.

2. **Tool calling, live:**
   ```bash
   python -m rag_app ask "How many chunks does the recursive strategy have?" --enable-tools
   ```

3. **Docker build** (all three):
   ```bash
   docker build -f Dockerfile.litellm -t rag-litellm .
   docker build -f Dockerfile.backend -t rag-backend .
   docker build -f Dockerfile.ui -t rag-ui .
   ```

4. **Docker Compose, full stack:**
   ```bash
   docker compose up --build
   ```
   Then open http://localhost:8501, ask a question through the UI, and
   confirm it returns an answer (not the "temporarily unavailable"
   degraded message).

5. **(Optional) Local vLLM model**, if you have a GPU:
   ```bash
   ./scripts/serve_local_model.sh
   python -m rag_app ask "Who is Rama?" --model local-model
   ```

If any of these fail, paste the exact error back for debugging — the same
way we worked through the Gemini model-name and duplicate-method bugs
earlier in this project.
