from .base import BaseChunker
from langchain_core.documents import Document as LCDocument
from typing import List, Dict, Any


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with character count and overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separator: str = "\n"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[LCDocument]:
        # Simple fixed-size chunking by character count
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(
                LCDocument(
                    page_content=chunk_text,
                    metadata={**metadata, "chunk_idx": len(chunks), "chunk_size": self.chunk_size, "strategy": "fixed"},
                )
            )
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
            if start >= len(text) and chunks:
                break
        return chunks