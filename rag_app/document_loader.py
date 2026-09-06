import os
from typing import List, Optional
from langchain_core.documents import Document as LCDocument


class RamayanaPDFLoader:
    """Loads Ramayana PDF and extracts text per page with metadata.

    Uses pdfplumber if available, falls back to basic file reading.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdfplumber_available = self._check_pdfplumber()

    def _check_pdfplumber(self):
        """Check if pdfplumber is available."""
        try:
            import pdfplumber
            return True
        except ImportError:
            return False

    def load(self) -> List[LCDocument]:
        """Extract text from PDF, one document per page."""
        if self.pdfplumber_available:
            return self._load_with_pdfplumber()
        else:
            return self._load_fallback()

    def _load_with_pdfplumber(self) -> List[LCDocument]:
        """Load PDF using pdfplumber."""
        import pdfplumber

        pages_docs = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    metadata = {
                        "page": i + 1,
                        "total_pages": len(pdf.pages),
                        "source": self.pdf_path,
                    }
                    doc = LCDocument(page_content=text, metadata=metadata)
                    pages_docs.append(doc)
        return pages_docs

    def _load_fallback(self) -> List[LCDocument]:
        """Fallback: read the PDF as text (basic approach)."""
        # This is a very basic fallback - just reads the file as binary
        # In a real setup, would use pypdf or similar
        docs = []
        try:
            with open(self.pdf_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            # Create a single document with the whole text
            metadata = {
                "page": 1,
                "total_pages": 1,
                "source": self.pdf_path,
                "fallback": True,
            }
            doc = LCDocument(page_content=text, metadata=metadata)
            docs.append(doc)
        except Exception as e:
            print(f"Fallback load failed: {e}")
        return docs

    def load_all_text(self) -> str:
        """Return full text of the PDF as a single string."""
        if self.pdfplumber_available:
            import pdfplumber

            full_text = []
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
            return "\n".join(full_text)
        else:
            # Fallback: just return empty or minimal
            return ""


def extract_chunks_from_text(text: str, page_num: int = 0) -> List[str]:
    """Simple helper to split text into rough sentence chunks by page."""
    import re
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    # Group into chunks of ~10 sentences
    chunk_size = 10
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i : i + chunk_size]) + "."
        chunks.append(chunk)
    return chunks