# Pixel RAG Backend

FastAPI backend for the Visual Engineering Document QA system.  
**100% free APIs** — CLIP runs locally, Gemini uses the free tier.

---

## Architecture

```
User Query (text)
      │
      ▼
CLIP Text Encoder  ──────────────────────────────────────────────┐
      │                                                           │
      ▼                               PDF Upload                 │
FAISS Index ◄── CLIP Image Encoder ◄── pdf2image (page tiles)   │
      │                                                           │
      ▼                                                           │
Top-K Page Images (retrieved)  ◄────────────────────────────────┘
      │
      ▼
Gemini 1.5 Flash (multimodal generation)
      │
      ▼
Grounded Answer + Source Pages
```

---

## Free APIs Used

| Component         | Tool                            | Cost  |
|-------------------|---------------------------------|-------|
| Vision Embedding  | CLIP (HuggingFace, local)       | Free  |
| Text Embedding    | CLIP text encoder (local)       | Free  |
| Vector DB         | FAISS (in-memory, local)        | Free  |
| PDF Rendering     | pdf2image + poppler (local)     | Free  |
| LLM Generation    | Google Gemini 1.5 Flash         | Free* |

\* Gemini free tier: 15 req/min, 1M tokens/min, 1,500 req/day

---

## Setup

### 1. System dependency — poppler

**Ubuntu / Debian:**
```bash
sudo apt-get install -y poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download from https://github.com/oschwartz10612/poppler-windows/releases  
Add the `bin/` folder to your PATH.

---

### 2. Python environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3. Environment variables

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
# Get a free key at: https://aistudio.google.com/app/apikey
```

---

### 4. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

---

## API Endpoints

### `GET /health`
Health check.

### `POST /upload`
Upload and index a PDF file.
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

### `POST /query`
Query the indexed documents.
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me the torque specifications for bolt assembly", "top_k": 3}'
```

Response:
```json
{
  "answer": "According to page 14 of the manual...",
  "sources": [
    {
      "doc_id": "engineering_manual",
      "page": 14,
      "score": 0.8423,
      "image_b64": "...",
      "metadata": {}
    }
  ],
  "query": "Show me the torque specifications..."
}
```

### `GET /status`
Check how many pages are indexed.

### `DELETE /index`
Clear the vector store.

---

## Project Structure

```
pixel_rag_backend/
├── main.py                    # FastAPI app + route handlers
├── config.py                  # Settings (env vars, model names)
├── requirements.txt
├── .env.example
├── README.md
└── services/
    ├── __init__.py
    ├── pdf_processor.py       # PDF → image tiles (pdf2image)
    ├── embedding_service.py   # CLIP embeddings (HuggingFace)
    ├── vector_store.py        # FAISS index + metadata store
    └── rag_pipeline.py        # Retrieval + Gemini generation
```

---

## Connect to your React Frontend

In your React app, set the base URL:
```javascript
const API_BASE = "http://localhost:8000";

// Upload
const res = await fetch(`${API_BASE}/upload`, {
  method: "POST",
  body: formData,  // FormData with the PDF file
});

// Query
const res = await fetch(`${API_BASE}/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: userQuery, top_k: 3 }),
});
const { answer, sources } = await res.json();
```
