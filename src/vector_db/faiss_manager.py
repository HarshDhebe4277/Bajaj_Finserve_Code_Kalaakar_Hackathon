# src/vector_db/faiss_manager.py

import faiss
import numpy as np
from typing import List, Dict, Any, Optional

class FAISSManager:
    _instance = None
    _index = None
    _texts = []  # Store original text chunks corresponding to FAISS indices
    _metadatas = []  # Store metadata for each chunk (e.g., source, page number)

    def __new__(cls, dimension: int = 384):
        """Ensures only one instance of FAISSManager is created (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(FAISSManager, cls).__new__(cls)
            print(f"Initializing FAISS index with dimension: {dimension}...")
            cls._index = faiss.IndexFlatL2(dimension)
            cls._texts = []
            cls._metadatas = []
            print("FAISS index initialized successfully.")
        return cls._instance

    def add_documents(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Adds document embeddings and their corresponding texts/metadatas to the FAISS index.
        """
        if not embeddings or not texts:
            print("No embeddings or texts to add.")
            return

        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings and texts must match.")

        embeddings_np = np.array(embeddings).astype("float32")
        self._index.add(embeddings_np)

        self._texts.extend(texts)
        if metadatas:
            if len(metadatas) != len(texts):
                raise ValueError("Number of metadatas and texts must match if provided.")
            self._metadatas.extend(metadatas)
        else:
            self._metadatas.extend([{} for _ in texts])

        print(f"Added {len(embeddings)} documents to FAISS index. Total documents: {self._index.ntotal}.")

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search in the FAISS index.
        """
        if self._index.ntotal == 0:
            print("FAISS index is empty. No search performed.")
            return []

        query_embedding_np = np.array([query_embedding]).astype("float32")
        distances, indices = self._index.search(query_embedding_np, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            text = self._texts[idx]
            metadata = self._metadatas[idx] if idx < len(self._metadatas) else {}
            distance = distances[0][i]
            results.append({
                "text": text,
                "metadata": metadata,
                "distance": float(distance)
            })
        return results

    def reset_index(self):
        """Resets the FAISS index and stored texts/metadatas."""
        self._index.reset()
        self._texts = []
        self._metadatas = []
        print("FAISS index reset.")