# src/llm/gemini_llm_client.py

import os
import google.generativeai as genai
from typing import List, Dict, Any

class GeminiLLMClient:
    _instance = None
    _client = None
    # We choose a suitable Gemini model. For chat, 'gemini-pro' is a good general choice.
    # For more advanced capabilities, consider 'gemini-1.5-pro-latest' if you have access.
    MODEL_NAME = "gemini-pro" 

    def __new__(cls):
        """Ensures only one instance of GeminiLLMClient is created (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(GeminiLLMClient, cls).__new__(cls)
            gemini_api_key = os.getenv("GOOGLE_API_KEY")  # Use Gemini API key
            if not gemini_api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            
            print("Initializing Gemini LLM client (Async)...")
            genai.configure(api_key=gemini_api_key) # Configure the genai library with the API key
            cls._client = genai.GenerativeModel(model_name=cls.MODEL_NAME) # Initialize the GenerativeModel
            print(f"Gemini LLM client initialized with model: {cls.MODEL_NAME}.")
        return cls._instance

    async def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generates an answer to a query using the Gemini LLM based on provided context chunks.
        
        Args:
            query (str): The user's question.
            context_chunks (List[str]): A list of retrieved text chunks from the document.

        Returns:
            str: The LLM's generated answer.
        """
        if not self._client:
            raise RuntimeError("Gemini LLM client not initialized.")

        # --- Prompt Engineering Strategy (CRITICAL for Accuracy and Token Efficiency) ---
        context_string = "\n\n".join(context_chunks)

        # Gemini's `generate_content` expects a list of parts, which can be strings.
        # The prompt is constructed to guide Gemini to act as an expert assistant.
        prompt = (
            "You are an expert assistant for Bajaj Finserv. "
            "Answer ONLY using the information provided in the document context below. "
            "If the answer is not explicitly present in the context, reply exactly: 'Information not found in the provided document.' "
            "Do NOT guess, do NOT use outside knowledge, and do NOT include section numbers or legal clauses unless directly quoted in the answer. "
            "Keep your answer factual, concise (1-2 sentences), and avoid unnecessary details. "
            "If a question requires conditions to be satisfied, clearly state those conditions based only on the context.\n\n"
            f"Context:\n{context_string}\n\n"
            f"Question: {query}\n\n"
            "Instructions: Answer in 1-2 sentences, using ONLY the context above. "
            "Do not add extra information or repeat the question. "
            "If the answer is not found, reply: 'Information not found in the provided document.'\n"
            "Answer:"
        )

        try:
            # For asynchronous calls with google.generativeai, you use `generate_content` directly.
            # The `safety_settings` are crucial for controlling Gemini's behavior.
            # `temperature` should be low for factual answers. `max_output_tokens` limits length.
            response = await self._client.generate_content_async( # Changed from chat.completions.create
                contents=[prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,  # Keep low for factual, deterministic answers
                    max_output_tokens=500, # Limit response length
                ),
                # It's highly recommended to configure safety settings based on your application's needs.
                # Here's a strict example to ensure no harmful content is generated.
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            
            # Extract the text content from the Gemini response
            if response.candidates:
                # Access the text from the first part of the first candidate
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return "LLM could not generate a coherent answer."

        except Exception as e:
            print(f"Error calling Gemini LLM: {e}")
            return f"Error processing query with LLM: {e}"