import numpy as np
from typing import List, Dict, Any


def rrf(rank_lists: List[List[int]], k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.

    Args:
        rank_lists: List of ranked document index lists from different retrievers
        k: RRF constant (default 60, as per standard implementation)

    Returns:
        Combined ranked document indices
    """
    score_sums = {}
    for rank_list in rank_lists:
        for rank, doc_idx in enumerate(rank_list):
            if doc_idx not in score_sums:
                score_sums[doc_idx] = 0
            score_sums[doc_idx] += 1 / (k + rank + 1)

    # Sort by fused score (descending)
    sorted_indices = sorted(score_sums.keys(), key=lambda x: score_sums[x], reverse=True)
    return sorted_indices


class HybridRetriever:
    """Combines dense (embedding-based) and sparse (BM25-like) retrieval.

    Strategy:
    1. Dense search: ChromaDB vector similarity or mock similarity
    2. Sparse search: Keyword-based search using simple scoring
    3. Fuse results using Reciprocal Rank Fusion (RRF)
    4. Return top-K fused results
    """

    def __init__(
        self,
        vector_store,
        bm25_corpus: List[str] = None,
        embedding_function=None,
        k_dense: int = 10,
        k_sparse: int = 10,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_corpus = bm25_corpus or []
        self.embedding_function = embedding_function
        self.k_dense = k_dense
        self.k_sparse = k_sparse
        self.rrf_k = rrf_k

    def _dense_search(self, query: str, strategy_name: str, k: int = None) -> List[Dict]:
        """Perform dense vector search in the strategy's collection."""
        if k is None:
            k = self.k_dense

        if self.embedding_function:
            try:
                query_embedding = self.embedding_function(query)
            except Exception:
                query_embedding = None
        else:
            query_embedding = None

        # Search the vector store
        if query_embedding is not None and len(query_embedding) > 0 and self.vector_store:
            try:
                results = self.vector_store.similarity_search(
                    strategy_name=strategy_name,
                    query=query,
                    embedding_function=self.embedding_function,
                    k=k,
                )
                # Convert to ranked format
                ranked = []
                for i, doc in enumerate(results):
                    ranked.append({
                        "doc_idx": i,
                        "document": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                        "score": 0.0,  # Score will be fused with RRF
                    })
                return ranked
            except Exception as e:
                # Log loudly instead of swallowing silently - a silent except here
                # previously masked a NameError in vector_store.py for the entire
                # dense-retrieval path, causing every result to fall back to
                # sparse search (which has no real per-chunk metadata).
                print(f"[WARNING] Dense search failed, falling back to sparse: {e}")

        # Fallback: use simple keyword-based ranking from corpus
        return self._sparse_search(query, k)

    def _sparse_search(self, query: str, k: int = None) -> List[Dict]:
        """Perform sparse keyword search using simple TF-IDF-like scoring."""
        if k is None:
            k = self.k_sparse

        if not self.bm25_corpus:
            return []

        tokenized_query = query.lower().split()

        # Simple BM25-esque scoring without the rank_bm25 library
        scores = []
        for doc_idx, document in enumerate(self.bm25_corpus):
            doc_tokens = document.lower().split()
            # Calculate simple overlap score
            overlap = len(set(tokenized_query) & set(doc_tokens))
            # Length normalization
            doc_len_norm = 1 / (len(doc_tokens) + 0.1) if doc_tokens else 0
            score = overlap * doc_len_norm
            scores.append((score, doc_idx))

        # Sort by score (descending) and take top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        ranked = []
        for score, doc_idx in scores[:k]:
            if score > 0:
                ranked.append({
                    "doc_idx": doc_idx,
                    "document": self.bm25_corpus[doc_idx],
                    "metadata": {},
                    "score": score,
                })

        return ranked

    def retrieve(self, query: str, strategy_name: str) -> List[Dict]:
        """Perform hybrid search: dense + sparse + RRF.

        Args:
            query: User query text
            strategy_name: Which chunking strategy collection to search

        Returns:
            List of documents with fused scores
        """
        # Dense search
        dense_ranked = self._dense_search(query, strategy_name, self.k_dense)

        # Sparse search
        sparse_ranked = self._sparse_search(query, self.k_sparse)

        # Combine ranks using RRF
        dense_indices = [item["doc_idx"] for item in dense_ranked]
        sparse_indices = [item["doc_idx"] for item in sparse_ranked]

        fused_rank = rrf([dense_indices, sparse_indices], self.rrf_k)

        # Build fused results
        corpus_len = len(self.bm25_corpus) if self.bm25_corpus else 100
        fused_results = []
        for rank, doc_idx in enumerate(fused_rank):
            if doc_idx < corpus_len:
                # Get the document info from dense or sparse results
                doc_info = None
                for item in dense_ranked:
                    if item["doc_idx"] == doc_idx:
                        doc_info = item
                        break
                if doc_info is None:
                    for item in sparse_ranked:
                        if item["doc_idx"] == doc_idx:
                            doc_info = item
                            break

                if doc_info:
                    fused_results.append({
                        "doc_idx": doc_idx,
                        "document": doc_info["document"],
                        "metadata": doc_info["metadata"],
                        "fused_score": 1 / (self.rrf_k + rank),
                        "rank": rank,
                    })

        return fused_results[: self.k_dense]


class CrossEncoderReranker:
    """Cross-encoder re-ranker for reranking retrieved passages.

    Uses a lightweight approach for re-ranking when cross-encoder models
    aren't available.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """Re-rank candidate documents.

        Args:
            query: The user's question
            candidates: List of candidate documents from hybrid search
            top_k: Number of top results to return

        Returns:
            Re-ranked list of documents
        """
        if top_k is None:
            top_k = min(len(candidates), 5)

        if not candidates:
            return []

        if self.model:
            try:
                # Use cross-encoder if available
                sentence_pairs = [[query, c.get("document", "")] for c in candidates]
                scores = self.model.predict(sentence_pairs)

                scored_candidates = []
                for i, candidate in enumerate(candidates):
                    scored_candidate = {
                        **candidate,
                        "rerank_score": float(scores[i]),
                    }
                    scored_candidates.append(scored_candidate)

                scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored_candidates[:top_k]
            except Exception:
                pass

        # Fallback: simple relevance scoring based on keyword overlap
        return self._fallback_rerank(query, candidates, top_k)

    def _fallback_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Fallback re-ranking using query-document overlap."""
        query_terms = set(query.lower().split())

        scored = []
        for candidate in candidates:
            doc_terms = set(candidate.get("document", "").lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / (len(query_terms) + 1e-5)
            scored.append({
                **candidate,
                "rerank_score": score,
            })

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]

    def rerank_with_documents(
        self,
        query: str,
        documents: List[Any],
        top_k: int = None,
    ) -> List[Any]:
        """Re-rank LangChain Document objects."""
        # Convert documents to dict format
        candidates = []
        for i, doc in enumerate(documents):
            candidates.append({
                "document": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                "original_idx": i,
            })

        ranked = self.rerank(query, candidates, top_k=top_k)

        # Return actual Document objects in new order
        result_docs = []
        for item in ranked:
            orig_idx = item.get("original_idx", -1)
            if orig_idx >= 0:
                result_docs.append(documents[orig_idx])
            else:
                # Create a simple object
                class _Doc:
                    pass
                d = _Doc()
                d.page_content = item.get("document", "")
                d.metadata = item.get("metadata", {})
                result_docs.append(d)

        return result_docs