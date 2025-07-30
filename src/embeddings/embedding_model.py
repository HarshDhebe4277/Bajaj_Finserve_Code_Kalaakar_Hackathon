# src/embeddings/embedding_model.py
import os
import requests
from typing import List

class EmbeddingModel:
    _instance = None
    MODEL_NAME = "nomic-embed-text-v1"
    NOMIC_API_URL = "https://api-atlas.nomic.ai/v1/embedding/text"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls.api_key = os.getenv("NOMIC_API_KEY")
            if not cls.api_key:
                raise ValueError(
                    "❌ NOMIC_API_KEY environment variable not set.\n"
                    "👉 Set it in your deployment environment."
                )
            print(f"Using Nomic API for embeddings: {cls.MODEL_NAME}")
        return cls._instance

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings using Nomic Atlas API (REST).
        """
        if not texts:
            return []

        payload = {
            "texts": texts,
            "model": self.MODEL_NAME,
            "task_type": "search_document"
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.NOMIC_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("embeddings", [])
        except Exception as e:
            print(f"Error generating embeddings via Nomic API: {e}")
            return []