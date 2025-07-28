# src/llm/groq_llm_client.py

import os
from groq import AsyncGroq # IMPORTANT: Use AsyncGroq for async operations
from typing import List, Dict, Any

class GroqLLMClient:
    _instance = None
    _client = None
    # We choose llama-3.1-8b-instant for its speed and efficiency on Groq
    MODEL_NAME = "llama-3.1-8b-instant" 

    def __new__(cls):
        """Ensures only one instance of GroqLLMClient is created (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(GroqLLMClient, cls).__new__(cls)
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            
            print("Initializing Groq LLM client (Async)...")
            cls._client = AsyncGroq(api_key=groq_api_key) # Initialize with AsyncGroq
            print(f"Groq LLM client initialized with model: {cls.MODEL_NAME}.")
        return cls._instance

    async def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generates an answer to a query using the Groq LLM based on provided context chunks.
        
        Args:
            query (str): The user's question.
            context_chunks (List[str]): A list of retrieved text chunks from the document.

        Returns:
            str: The LLM's generated answer.
        """
        if not self._client:
            raise RuntimeError("Groq LLM client not initialized.")

        # --- Prompt Engineering Strategy (CRITICAL for Accuracy and Token Efficiency) ---
        context_string = "\n\n".join(context_chunks)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly accurate and concise assistant for Bajaj Finserv, specialized "
                    "in providing answers based ONLY on the provided document context. "
                    "If the answer to the question cannot be found directly within the provided context, "
                    "you MUST state 'Information not found in the provided document.' "
                    "Do not make up information. Prioritize factual accuracy and brevity. "
                    "if some query ansers requires some conditions to be satisfied for that query also give the conditions to satisfy"
                    "Give a concise (1-2 sentences) answer using only the provided context. Include only the key information needed to answer the question. Do NOT include section numbers, legal clauses, or extra text unless they are essential."
 # Strengthened instruction for detail
                )
            },
            {
                "role": "user",
                "content": (
                    f"Answer the question in 1-2 sentences, using only the context below. "
                    f"Do not repeat unnecessary details or section numbers.\n\n"
                    f"--- DOCUMENT CONTEXT ---\n{context_string}\n\n"
                    f"--- QUESTION ---\n{query}\n\n"
                    f"Your concise answer:"

                )
            }
        ]

        try:
            # The await here is correct, as chat.completions.create is an awaitable method
            # when the client is initialized with AsyncGroq.
            chat_completion = await self._client.chat.completions.create( 
                messages=messages,
                model=self.MODEL_NAME,
                temperature=0.0,  # Keep low for factual, deterministic answers
                max_tokens=500,   # Limit response length to save tokens and improve latency
                top_p=1,
                stop=None,
                stream=False, # We don't need streaming for this use case
            )
            # Extract the content from the LLM's response
            if chat_completion.choices and chat_completion.choices[0].message.content:
                return chat_completion.choices[0].message.content.strip()
            else:
                return "LLM could not generate a coherent answer."

        except Exception as e:
            print(f"Error calling Groq LLM: {e}")
            return f"Error processing query with LLM: {e}"