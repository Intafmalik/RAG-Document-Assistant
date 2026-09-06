from abc import ABC, abstractmethod
from langchain_core.documents import Document as LCDocument
from typing import List, Dict, Any


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[LCDocument]:
        """Split text into chunks with associated metadata.

        Args:
            text: The full text to chunk
            metadata: Base metadata (page, source, etc.)

        Returns:
            List of LangChain Document objects, each with chunk text and metadata
        """
        pass