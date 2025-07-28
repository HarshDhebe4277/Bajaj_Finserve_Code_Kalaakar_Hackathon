# src/embeddings/embedding_model.py

import os
from typing import List
from nomic import embed

class EmbeddingModel:
    _instance = None
    MODEL_NAME = "nomic-embed-text-v1"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            nomic_api_key = os.getenv("NOMIC_API_KEY")
            if not nomic_api_key:
                raise ValueError(
                    "NOMIC_API_KEY environment variable not set.\n"
                    "👉 Run 'nomic login <your_api_key>' or set NOMIC_API_KEY in your .env file."
                )
            print(f"Using Nomic API for embeddings: {cls.MODEL_NAME}")
        return cls._instance

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text strings using Nomic Embed API.
        """
        if not texts:
            return []

        try:
            response = embed.text(texts, model=self.MODEL_NAME)

            # Newer Nomic versions return dict with "embeddings" key
            if isinstance(response, dict) and "embeddings" in response:
                return response["embeddings"]

            # Some versions return an object with attribute .embeddings
            if hasattr(response, "embeddings"):
                return [e.embedding for e in response.embeddings]

            raise RuntimeError(f"Unexpected Nomic API response format: {response}")
        except Exception as e:
            print(f"Error generating embeddings via Nomic: {e}")
            return []
