# main.py
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
import os
import asyncio
import time

from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool

# Import utility functions
from src.utils.document_loader import extract_text_from_document
from src.utils.text_splitter import split_text_into_chunks
from src.embeddings.embedding_model import EmbeddingModel
from src.vector_db.faiss_manager import FAISSManager
from src.llm.groq_llm_client import GroqLLMClient

# Load environment variables
load_dotenv()

app = FastAPI(
    title="HackRx LLM-Powered Query Retrieval System",
    description="An intelligent system for processing documents and answering contextual queries.",
    version="1.0.0"
)

# --- Global components ---
startup_start_time = time.time()
embedding_generator = EmbeddingModel()
faiss_manager = FAISSManager(dimension=768)  # Nomic embedding dimension
groq_llm_client = GroqLLMClient()
startup_end_time = time.time()
print(f"Startup completed in {startup_end_time - startup_start_time:.2f}s.")

# --- Config ---
REQUIRED_AUTH_TOKEN = os.getenv("HACKRX_AUTH_TOKEN")

# --- Document Cache ---
# Using a simple dictionary as an in-memory cache for processed documents.
# Key: document_url, Value: Tuple[List[str], List[List[float]]] (text_chunks, chunk_embeddings)
document_cache: Dict[str, Tuple[List[str], List[List[float]]]] = {}

# --- Pydantic Models ---
class RunRequest(BaseModel):
    documents: str  # URL to the document blob
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run_hackrx_submission(request: Request, payload: RunRequest):
    # Auth check
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    if auth_header.split(" ")[1] != REQUIRED_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    document_url = payload.documents

    text_chunks: List[str] = []
    chunk_embeddings: List[List[float]] = []

    # --- Step 1 & 2 & 3 (Document Processing with Caching) ---
    # Check if the document has already been processed and is in cache
    if document_url in document_cache:
        print(f"Retrieving processed document from cache: {document_url}")
        text_chunks, chunk_embeddings = document_cache[document_url]
    else:
        # If not in cache, proceed with full document processing
        print(f"Processing new document: {document_url}")
        # Step 1: Extract Text
        document_text = await run_in_threadpool(extract_text_from_document, document_url)
        if not document_text.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Unable to extract text from document.")

        # Step 2: Chunking
        # Adjusted CHUNK_SIZE and CHUNK_OVERLAP for more granular chunks
        CHUNK_SIZE = 500  # Changed from 1000
        CHUNK_OVERLAP = 100 # Changed from 200
        text_chunks = await run_in_threadpool(
            split_text_into_chunks, document_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        print(f"Document split into {len(text_chunks)} chunks.")

        # Step 3: Embeddings
        print(f"Generating embeddings for {len(text_chunks)} chunks...")
        chunk_embeddings = await run_in_threadpool(embedding_generator.get_embeddings, text_chunks)
        if not chunk_embeddings:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Embedding generation failed.")

        # Store the processed document data in cache
        document_cache[document_url] = (text_chunks, chunk_embeddings)
        print(f"Document {document_url} processed and cached.")

    # --- Populate FAISS for current request ---
    # The FAISS index is a singleton and must be reset and re-populated for each request
    # to ensure it only contains the relevant document's embeddings for the current query batch.
    faiss_manager.reset_index()
    await run_in_threadpool(faiss_manager.add_documents, chunk_embeddings, text_chunks)
    print(f"FAISS index built with {faiss_manager._index.ntotal} chunks for this request.")

    # --- Step 4: Answer Questions ---
    TOP_K_RETRIEVAL = 10 # Reverted to 10

    llm_tasks = []

    # Embed all questions in one go
    print(f"\nGenerating embeddings for {len(payload.questions)} questions...")
    all_query_embeddings = await run_in_threadpool(embedding_generator.get_embeddings, payload.questions)

    if not all_query_embeddings or len(all_query_embeddings) != len(payload.questions):
        for question in payload.questions:
            llm_tasks.append(asyncio.create_task(
                asyncio.sleep(0, result=f"Could not embed question (batch failed): {question}")
            ))
        answers = await asyncio.gather(*llm_tasks)
        return RunResponse(answers=answers)

    # Iterate through questions and their corresponding batch embeddings
    for i, question in enumerate(payload.questions):
        print(f"\nQuestion: {question}")
        query_embedding_single = all_query_embeddings[i]

        search_results = await run_in_threadpool(faiss_manager.search, query_embedding_single, k=TOP_K_RETRIEVAL)
        retrieved_contexts = [r["text"] for r in search_results] if search_results else []

        # --- IMPORTANT: Log retrieved contexts for debugging ---
        print(f"Retrieved contexts for question '{question}':")
        for j, context in enumerate(retrieved_contexts):
            print(f"  Chunk {j+1}: {context[:200]}...") # Print first 200 chars to keep logs readable
        # --- End Logging ---

        if not retrieved_contexts:
            llm_tasks.append(asyncio.create_task(
                asyncio.sleep(0, result=f"No relevant context found for question: {question}")
            ))
            continue

        llm_tasks.append(asyncio.create_task(groq_llm_client.generate_answer(question, retrieved_contexts)))

    answers = await asyncio.gather(*llm_tasks)
    return RunResponse(answers=answers)

# --- Health Check ---
@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "message": "API is running!"}
