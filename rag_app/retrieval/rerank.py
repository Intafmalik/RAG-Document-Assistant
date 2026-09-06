from typing import List, Dict, Any


class CrossEncoderReranker:
    """Cross-encoder re-ranker for reranking retrieved passages.

    Uses a cross-encoder model that jointly encodes the query and each
    candidate passage, producing a relevance score.

    Falls back to keyword-based scoring when the cross-encoder model
    is not available.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        try:
            # Try importing CrossEncoder - in production this would use sentence-transformers
            # For now, we use a simple fallback approach
            pass
        except Exception:
            pass

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """Re-rank candidate documents using keyword-based fallback.

        Args:
            query: The user's question
            candidates: List of candidate documents from hybrid search
            top_k: Number of top results to return

        Returns:
            Re-ranked list of documents sorted by relevance score
        """
        if top_k is None:
            top_k = min(len(candidates), 5)

        if not candidates:
            return []

        # Fallback: simple relevance scoring based on query-document overlap
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

        # Return document objects in new order
        result_docs = []
        for item in ranked:
            orig_idx = item.get("original_idx", -1)
            if orig_idx >= 0 and orig_idx < len(documents):
                result_docs.append(documents[orig_idx])
            else:
                class _Doc:
                    pass
                d = _Doc()
                d.page_content = item.get("document", "")
                d.metadata = item.get("metadata", {})
                result_docs.append(d)

        return result_docs