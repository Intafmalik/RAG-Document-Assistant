from .base import BaseChunker
from langchain_core.documents import Document as LCDocument
from typing import List, Dict, Any
import re


class StructuralChunker(BaseChunker):
    """Document-structure aware chunking for PDFs.

    Strategy: Split by PDF pages first, then detect chapter/section boundaries
    within each page text. This preserves the natural hierarchy of the document.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Common chapter markers in Indian texts
        self.chapter_patterns = [
            r"^Chapter\s+\d+",
            r"^Sarga\s+\d+",
            r"^Adhyaya\s+\d+",
        ]

    def _detect_structure(self, text: str) -> Dict[str, Any]:
        """Detect chapter/section headers in text."""
        lines = text.split("\n")
        headers = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            for pattern in self.chapter_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    headers.append({"line": line_stripped, "line_idx": i})
                    break
        return {"headers": headers, "total_lines": len(lines)}

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[LCDocument]:
        # If text is small enough, keep as single chunk preserving structure
        if len(text) <= self.chunk_size:
            return [
                LCDocument(
                    page_content=text,
                    metadata={**metadata, "chunk_idx": 0, "chunk_size": len(text), "strategy": "structural"},
                )
            ]

        # Split by paragraphs (double newlines)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        docs = []
        current_chunk = ""
        current_idx = metadata.get("chunk_idx", 0)

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + para
            else:
                # Save current chunk
                if current_chunk:
                    docs.append(
                        LCDocument(
                            page_content=current_chunk,
                            metadata={**metadata, "chunk_idx": current_idx, "chunk_size": len(current_chunk), "strategy": "structural"},
                        )
                    )
                    current_idx += 1
                current_chunk = para

        # Don't forget the last chunk
        if current_chunk:
            docs.append(
                LCDocument(
                    page_content=current_chunk,
                    metadata={**metadata, "chunk_idx": current_idx, "chunk_size": len(current_chunk), "strategy": "structural"},
                )
            )

        return docs