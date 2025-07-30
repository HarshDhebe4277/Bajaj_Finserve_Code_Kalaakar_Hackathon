# src/llm/groq_llm_client.py

import os
from groq import AsyncGroq # IMPORTANT: Use AsyncGroq for async operations
from typing import List, Dict, Any

class GroqLLMClient:
    _instance = None
    _client = None
    # We choose llama-3.1-8b-instant for its speed and efficiency on Groq
    MODEL_NAME = "gemma2-9b-it" 

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

        context_string = "\n\n".join(context_chunks)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly accurate and concise assistant for Bajaj Finserv, specialized "
                    "in providing answers based ONLY on the provided document context. "
                    "If the answer to the question cannot be found directly within the provided context, "
                    "you MUST state 'Information not found in the provided document.' "
                    "Do not make up information. Prioritize factual accuracy and extreme brevity. "
                    "Provide only the essential information needed to directly answer the question, in 1-2 sentences. "
                    "Do NOT include section numbers, legal clauses, or any extra text unless they are the direct and sole answer to the question itself. "
                    "If the answer requires specific conditions, list only the core, essential conditions directly related to the query."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Answer the question using only the context below. Focus strictly on the question asked.\n\n"
                    f"--- DOCUMENT CONTEXT ---\n{context_string}\n\n"
                    f"--- QUESTION ---\n{query}\n\n"
                    f"Your concise answer:"
                )
            }
        ]

        try:
            chat_completion = await self._client.chat.completions.create( 
                messages=messages,
                model=self.MODEL_NAME,
                temperature=0.0,  # Keep low for factual, deterministic answers
                max_tokens=250,   # Further limit response length to enforce brevity
                top_p=1,
                stop=None,
                stream=False,
            )
            if chat_completion.choices and chat_completion.choices[0].message.content:
                return chat_completion.choices[0].message.content.strip()
            else:
                return "LLM could not generate a coherent answer."

        except Exception as e:
            print(f"Error calling Groq LLM: {e}")
            return f"Error processing query with LLM: {e}"