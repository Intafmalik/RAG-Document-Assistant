from .base import BaseChunker
from langchain_core.documents import Document as LCDocument
from typing import List, Dict, Any
import re


class SemanticChunker(BaseChunker):
    """Semantic chunking using embeddings to find natural boundaries.

    Splits text based on similarity of sentence embeddings.
    When similarity drops below a threshold, a new chunk starts.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedding_model_name)
        except ImportError:
            pass

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[LCDocument]:
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            if text.strip():
                return [
                    LCDocument(
                        page_content=text,
                        metadata={**metadata, "chunk_idx": 0, "chunk_size": self.chunk_size, "strategy": "semantic"},
                    )
                ]
            return []

        if self.embedder is None:
            # Fallback: just split by chunk_size
            return self._fallback_chunk(text, sentences, metadata)

        # Get embeddings for sentences
        embeddings = self.embedder.encode(sentences)

        # Compute cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            dot = sum(a * b for a, b in zip(embeddings[i], embeddings[i + 1]))
            norm_a = sum(a * a for a in embeddings[i]) ** 0.5
            norm_b = sum(b * b for b in embeddings[i + 1]) ** 0.5
            if norm_a > 0 and norm_b > 0:
                sim = dot / (norm_a * norm_b)
            else:
                sim = 0
            similarities.append(sim)

        # Build chunks based on similarity thresholds
        threshold = 0.5
        chunk_boundaries = [0]
        for i in range(1, len(sentences)):
            if similarities[i - 1] < threshold:
                chunk_boundaries.append(i)
        chunk_boundaries.append(len(sentences))

        chunks = []
        for i in range(len(chunk_boundaries) - 1):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i + 1]
            chunk_text = ". ".join(sentences[start:end])
            if chunk_text.strip():
                chunks.append(
                    LCDocument(
                        page_content=chunk_text.strip(),
                        metadata={**metadata, "chunk_idx": i, "chunk_size": self.chunk_size, "strategy": "semantic"},
                    )
                )

        return chunks

    def _fallback_chunk(self, text: str, sentences: List[str], metadata: Dict[str, Any]) -> List[LCDocument]:
        """Fallback chunking when embeddings aren't available."""
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(
                        LCDocument(
                            page_content=current_chunk,
                            metadata={**metadata, "chunk_idx": len(chunks), "chunk_size": self.chunk_size, "strategy": "semantic"},
                        )
                    )
                current_chunk = sentence
        if current_chunk:
            chunks.append(
                LCDocument(
                    page_content=current_chunk,
                    metadata={**metadata, "chunk_idx": len(chunks), "chunk_size": self.chunk_size, "strategy": "semantic"},
                )
            )
        return chunks