import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


class VectorStoreManager:
    """Manages vector storage for different chunking strategies.

    Note: This implementation uses a mock/store-based approach when chromadb
    is not available. When chromadb is installed, it provides persistent
    vector storage with similarity search.
    """

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or "./chroma_db"
        self.collections: Dict[str, Any] = {}
        self.doc_store: Dict[str, List[Any]] = {}  # In-memory document store
        self.embedding_store: Dict[str, List[List[float]]] = {}

        os.makedirs(self.persist_dir, exist_ok=True)

        # Try to initialize chroma, but don't fail if not available
        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self._use_chroma = True
        except ImportError:
            self.client = None
            self._use_chroma = False
            print("[INFO] chromadb not available - using in-memory store")

    def get_collection(self, strategy_name: str):
        """Get or create a ChromaDB collection for the given strategy."""
        if strategy_name in self.collections:
            return self.collections[strategy_name]

        if self._use_chroma:
            try:
                collection = self.client.get_collection(name=f"ramayana_{strategy_name}")
                self.collections[strategy_name] = collection
                return collection
            except Exception:
                collection = self.client.create_collection(
                    name=f"ramayana_{strategy_name}",
                    metadata={"strategy": strategy_name},
                )
                self.collections[strategy_name] = collection
                return collection
        else:
            # Mock collection
            if strategy_name not in self.collections:
                self.collections[strategy_name] = {"docs": [], "ids": [], "embeddings": []}
            return self.collections[strategy_name]

    def add_documents(
        self,
        strategy_name: str,
        documents: List[Any],
        embedding_function,
    ) -> None:
        """Add chunked documents to the collection for a given strategy."""
        collection = self.get_collection(strategy_name)

        if self._use_chroma:
            ids = []
            metadatas = []
            texts = []
            embeddings = []

            for i, doc in enumerate(documents):
                doc_id = f"{strategy_name}_{i}" # Use current index for unique ID
                ids.append(doc_id)
                metadatas.append(doc.metadata)
                texts.append(doc.page_content)
                embeddings.append(embedding_function(doc.page_content).tolist() if embedding_function else None)

            # Filter out None embeddings if no embedding function is provided but Chroma is active
            if None in embeddings and embedding_function is None:
                print("[WARNING] ChromaDB active but no embedding function provided. Skipping embedding storage.")
                embeddings = None # Let ChromaDB use its default embedding function if available

            max_batch_size = 5000 # ChromaDB's default max batch size or slightly less for safety
            for i in range(0, len(ids), max_batch_size):
                batch_ids = ids[i:i + max_batch_size]
                batch_embeddings = embeddings[i:i + max_batch_size] if embeddings is not None else None
                batch_texts = texts[i:i + max_batch_size]
                batch_metadatas = metadatas[i:i + max_batch_size]

                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                )
        else: # In-memory fallback
            for doc in documents:
                doc_id = f"{strategy_name}_{doc.metadata.get('chunk_idx', 0)}"
                embedding = embedding_function(doc.page_content) if embedding_function else [0.0] * 384
                metadata = doc.metadata

                collection["docs"].append(doc.page_content)
                collection["ids"].append(doc_id)
                collection["embeddings"].append(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)

        print(f"Added {len(documents)} documents to strategy: {strategy_name}")

    def similarity_search(
        self,
        strategy_name: str,
        query: str,
        embedding_function,
        k: int = 10,
    ) -> List[Any]:
        """Perform dense vector similarity search in a strategy collection."""
        collection = self.get_collection(strategy_name)

        if self._use_chroma:
            query_embedding = embedding_function(query).tolist() if embedding_function else None

            if query_embedding is None:
                print("[WARNING] ChromaDB active but no embedding function provided for query. Cannot perform similarity search.")
                return []
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            docs = []
            if results["documents"] and results["documents"][0]:
                for j in range(len(results["documents"][0])):
                    doc = Document(
                        page_content=results["documents"][0][j],
                        metadata=results["metadatas"][0][j] if results["metadatas"] else {},
                    )
                    docs.append(doc)
            return docs
        else: # In-memory fallback
            if not collection["docs"]:
                return []

            query_embedding = embedding_function(query) if embedding_function else [0.0] * 384

            # Simple cosine similarity search
            scores = []
            for i, (doc_content, emb) in enumerate(zip(collection["docs"], collection["embeddings"])):
                # Compute cosine similarity
                dot = sum(a * b for a, b in zip(query_embedding, emb))
                norm_a = sum(a * a for a in query_embedding) ** 0.5
                norm_b = sum(b * b for b in emb) ** 0.5
                if norm_a > 0 and norm_b > 0:
                    sim = dot / (norm_a * norm_b)
                else:
                    sim = 0
                scores.append((sim, i))

            # Sort by similarity (descending) and take top-k
            scores.sort(key=lambda x: x[0], reverse=True)
            results = []
            for sim, idx in scores[:k]:
                if sim > 0: # Only return if there's some similarity
                    # Need to retrieve original metadata if available
                    metadata_from_collection = collection["metadatas"][idx] if "metadatas" in collection and len(collection["metadatas"]) > idx else {}
                    results.append(
                        Document(page_content=collection["docs"][idx], metadata=metadata_from_collection)
                    )
            return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about all collections."""
        stats = {}
        if self._use_chroma:
            # List all collections from ChromaDB directly
            try:
                collections = self.client.list_collections()
                for collection in collections:
                    name = collection.name
                    count = collection.count()
                    stats[name] = {"document_count": count, "metadata": collection.metadata}
            except Exception as e:
                print(f"Error listing collections: {e}")
        else:
            for name in list(self.collections.keys()):
                collection = self.collections[name]
                count = len(collection.get("docs", []))
                stats[name] = {"document_count": count}
        return stats

    def delete_collection(self, strategy_name: str) -> None:
        """Delete a collection for a given strategy."""
        if strategy_name in self.collections:
            if self._use_chroma:
                try:
                    self.client.delete_collection(name=f"ramayana_{strategy_name}")
                    print(f"Deleted ChromaDB collection: ramayana_{strategy_name}")
                except Exception as e:
                    print(f"Error deleting ChromaDB collection ramayana_{strategy_name}: {e}")
            
            del self.collections[strategy_name]
            print(f"Deleted in-memory collection entry: {strategy_name}")

    def rebuild_strategy_collection(
        self,
        strategy_name: str,
        documents: List[Any],
        embedding_function,
    ) -> None:
        """Rebuild a collection: clear and re-add all documents."""
        self.delete_collection(strategy_name)
        self.add_documents(strategy_name, documents, embedding_function)

    def __del__(self):
        """Cleanup."""
        pass