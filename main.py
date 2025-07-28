# main.py
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
from typing import List
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

    # --- Step 1: Extract Text ---
    print(f"Extracting text from: {document_url}")
    document_text = await run_in_threadpool(extract_text_from_document, document_url)
    if not document_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Unable to extract text from document.")

    # --- Step 2: Chunking ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    text_chunks = await run_in_threadpool(
        split_text_into_chunks, document_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    print(f"Document split into {len(text_chunks)} chunks.")

    # --- Step 3: Embeddings & FAISS ---
    print(f"Generating embeddings for {len(text_chunks)} chunks...")
    chunk_embeddings = await run_in_threadpool(embedding_generator.get_embeddings, text_chunks)
    if not chunk_embeddings:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Embedding generation failed.")

    faiss_manager.reset_index()
    await run_in_threadpool(faiss_manager.add_documents, chunk_embeddings, text_chunks)
    print(f"FAISS index built with {faiss_manager._index.ntotal} chunks.")

    # --- Step 4: Answer Questions ---
    TOP_K_RETRIEVAL = 10
    llm_tasks = []

    for question in payload.questions:
        print(f"\nQuestion: {question}")
        query_embedding = await run_in_threadpool(embedding_generator.get_embeddings, [question])
        if not query_embedding:
            llm_tasks.append(asyncio.create_task(
                asyncio.sleep(0, result=f"Could not embed question: {question}")
            ))
            continue
        query_embedding_single = query_embedding[0]

        search_results = await run_in_threadpool(faiss_manager.search, query_embedding_single, k=TOP_K_RETRIEVAL)
        retrieved_contexts = [r["text"] for r in search_results] if search_results else []

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
