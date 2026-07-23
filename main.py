"""
Pixel RAG Backend - FastAPI Application
Uses: HuggingFace CLIP (free) for embeddings + Google Gemini (free) for generation
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from services.pdf_processor import PDFProcessor
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.rag_pipeline import RAGPipeline
from config import settings

app = FastAPI(
    title="Pixel RAG API",
    description="Visual document QA using Pixel RAG (CLIP embeddings + Gemini generation)",
    version="1.0.0"
)

# CORS — allow all origins for local dev; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton services (loaded once at startup) ────────────────────────────────
pdf_processor = PDFProcessor()
embedding_service = EmbeddingService()
vector_store = VectorStore()
rag_pipeline = RAGPipeline(embedding_service, vector_store)


# ── Request / Response schemas ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    document_id: Optional[str] = None   # filter to a single uploaded doc


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str


class StatusResponse(BaseModel):
    status: str
    total_pages_indexed: int
    documents: list[str]


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "message": "Pixel RAG backend is running"}


# ── Upload & index a PDF ───────────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    1. Accept a PDF upload.
    2. Convert every page to an image tile.
    3. Embed each tile with CLIP.
    4. Store vectors in the in-memory FAISS index.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        pdf_bytes = await file.read()
        doc_id = file.filename.replace(" ", "_").replace(".pdf", "")

        # Convert pages → PIL images
        pages = pdf_processor.pdf_to_images(pdf_bytes, doc_id=doc_id)

        # Embed + index
        indexed = await embedding_service.embed_and_index(pages, vector_store)

        return {
            "success": True,
            "document_id": doc_id,
            "pages_indexed": indexed,
            "message": f"Successfully indexed {indexed} pages from '{file.filename}'"
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Query the RAG pipeline ─────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    1. Embed the text query with CLIP (text encoder).
    2. Retrieve top-K visually similar page tiles.
    3. Send tiles + query to Gemini for a grounded answer.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if vector_store.total_vectors() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please upload a PDF first."
        )

    try:
        result = await rag_pipeline.run(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id
        )
        return QueryResponse(**result)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Index status ───────────────────────────────────────────────────────────────
@app.get("/status", response_model=StatusResponse)
async def status():
    return StatusResponse(
        status="ready",
        total_pages_indexed=vector_store.total_vectors(),
        documents=vector_store.list_documents()
    )


# ── Clear index ────────────────────────────────────────────────────────────────
@app.delete("/index")
async def clear_index():
    vector_store.clear()
    return {"success": True, "message": "Index cleared."}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
