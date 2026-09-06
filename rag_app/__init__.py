"""RAG Application for Ramayana document queries.

Demonstrates:
- Chunking strategies (fixed, recursive, semantic, structural)
- Vector database creation, insertion, and retrieval (ChromaDB)
- Hybrid search (dense + sparse + RRF)
- Re-ranking (cross-encoder)
- Question answering via LiteLLM proxy (Vertex AI / Gemini)
"""
