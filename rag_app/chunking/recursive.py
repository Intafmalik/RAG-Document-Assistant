from .base import BaseChunker
from langchain_core.documents import Document as LCDocument
from typing import List, Dict, Any
import re


class RecursiveChunker(BaseChunker):
    """Recursive character splitting - hierarchically splits by separators.

    Strategy: paragraphs -> sentences -> words -> characters
    This is the most commonly used chunking strategy.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", "。", " ", ""]

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[LCDocument]:
        # Use recursive splitting with the defined separators
        return self._recursive_split(text, self.separators, metadata, 0)

    def _recursive_split(self, text: str, separators: List[str], metadata: Dict[str, Any], depth: int) -> List[LCDocument]:
        """Recursively split text by separators."""
        if len(text) <= self.chunk_size or depth >= len(separators) - 1:
            if text.strip():
                return [
                    LCDocument(
                        page_content=text.strip(),
                        metadata={**metadata, "chunk_idx": 0, "chunk_size": self.chunk_size, "strategy": "recursive"},
                    )
                ]
            return []

        separator = separators[depth]
        parts = re.split(f"({separator})", text)
        # Reconstruct the split parts (keeping separators with following text)
        parts_list = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i + 1]:
                parts_list.append(parts[i] + parts[i + 1])
                i += 2
            else:
                parts_list.append(parts[i])
                i += 1

        chunks = []
        current_chunk = ""
        for part in parts_list:
            part = part.strip()
            if not part:
                continue
            test_chunk = current_chunk + (" " if current_chunk else "") + part if current_chunk else part
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(
                        LCDocument(
                            page_content=current_chunk,
                            metadata={**metadata, "chunk_idx": len(chunks), "chunk_size": self.chunk_size, "strategy": "recursive"},
                        )
                    )
                current_chunk = part

        if current_chunk:
            chunks.append(
                LCDocument(
                    page_content=current_chunk,
                    metadata={**metadata, "chunk_idx": len(chunks), "chunk_size": self.chunk_size, "strategy": "recursive"},
                )
            )

        # Recursively split each part
        remaining_chunks = []
        for part in parts_list:
            part_stripped = part.strip()
            if part_stripped:
                recursive = self._recursive_split(part_stripped, separators, metadata, depth + 1)
                remaining_chunks.extend(recursive)

        # Adjust chunk indices for recursively split chunks
        for i, chunk in enumerate(remaining_chunks):
            chunk.metadata["chunk_idx"] = len(chunks) + i

        chunks.extend(remaining_chunks)
        return chunks