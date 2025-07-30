# src/embeddings/embedding_model.py
import os
import requests
from typing import List

class EmbeddingModel:
    _instance = None
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Change if needed
    HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_NAME}"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls.api_key = os.getenv("HF_TOKEN")
            if not cls.api_key:
                raise ValueError(
                    "❌ HF_TOKEN environment variable not set.\n"
                    "👉 Set it in your deployment environment."
                )
            print(f"Using Hugging Face API for embeddings: {cls.MODEL_NAME}")
        return cls._instance

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings using Hugging Face Inference API (REST).
        Each text is embedded individually due to API constraints.
        """
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        embeddings = []
        for text in texts:
            payload = {"inputs": text}
            try:
                response = requests.post(self.HF_API_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                # Use the mean pooling of token embeddings if the model returns token-level embeddings
                if isinstance(data, list) and isinstance(data[0], list):
                    # Mean pooling across token embeddings
                    pooled = [sum(col) / len(col) for col in zip(*data)]
                    embeddings.append(pooled)
                else:
                    embeddings.append(data)
            except Exception as e:
                print(f"Error generating embedding for text: '{text}'\n{e}")
                embeddings.append([])  # Fallback to empty embedding

        return embeddings
