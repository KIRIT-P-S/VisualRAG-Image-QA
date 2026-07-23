"""
RAG Pipeline
------------
1. Embed text query with CLIP
2. Retrieve top-K page tiles from FAISS
3. Send HIGH-RES page images to Gemini Vision for accurate reading
4. Return grounded answer + sources (with small thumbnail b64 for frontend)
"""

import asyncio
from typing import Optional

import google.generativeai as genai
from PIL import Image
import io, base64

from config import settings
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


class RAGPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self._gemini_configured = False

    def _setup_gemini(self):
        if not self._gemini_configured:
            if not settings.GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY not set. "
                    "Get a free key at https://aistudio.google.com/app/apikey"
                )
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._gemini_configured = True

    async def run(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        document_id: Optional[str] = None,
    ) -> dict:
        loop = asyncio.get_event_loop()

        # Step 1: Embed query with CLIP text encoder
        query_vec = await loop.run_in_executor(
            None, self.embedding_service.embed_text, query
        )

        # Step 2: Retrieve top-K visually similar pages from FAISS
        retrieved = self.vector_store.search(
            query_vector=query_vec,
            top_k=top_k,
            document_id=document_id,
        )

        if not retrieved:
            return {
                "answer": "No relevant pages found. Please upload a document first.",
                "sources": [],
                "query": query,
            }

        # Step 3: Generate answer using HIGH-RES images
        answer = await loop.run_in_executor(
            None, self._generate_with_gemini, query, retrieved
        )

        # Step 4: Return sources with SMALL thumbnail b64 for frontend display
        sources = [
            {
                "doc_id": r["doc_id"],
                "page": r["page"],
                "score": round(r["score"], 4),
                "image_b64": r["image_b64"],        # small — shown in chat
                "metadata": r["metadata"],
            }
            for r in retrieved
        ]

        return {"answer": answer, "sources": sources, "query": query}

    def _generate_with_gemini(self, query: str, retrieved: list[dict]) -> str:
        """
        Send the query + HIGH-RES page images to Gemini.
        Uses hires_b64 (300 DPI, up to 1600px) so chip labels and text are readable.
        """
        self._setup_gemini()
        model = genai.GenerativeModel(settings.GEMINI_MODEL)

        parts = [
            "You are a precise technical document analyst.\n\n"
            f"USER QUESTION: {query}\n\n"
            "=== READING RULES — FOLLOW STRICTLY ===\n"
            "1. Read ONLY what is physically printed in the image. No prior knowledge.\n"
            "2. For component/chip names: copy the EXACT text from inside each IC box.\n"
            "3. If you see 'ATMEGA8', write 'ATMEGA8' — not 'ATMEGA328P'.\n"
            "4. If you see 'ATMEGA8U2-MU', write 'ATMEGA8U2-MU' — not 'ATMEGA16U2'.\n"
            "5. If text is too small to read, say 'text unclear' — do not guess.\n"
            "6. Start your answer with: 'Reading directly from the image:'\n\n"
            f"The {len(retrieved)} most relevant page(s) are shown below:\n\n"
        ]

        for result in retrieved:
            # Use hires_b64 if available, fall back to image_b64
            b64_to_use = result.get("hires_b64") or result["image_b64"]
            pil_img = _b64_to_pil(b64_to_use)

            parts.append(
                f"--- Page {result['page']} from '{result['doc_id']}' "
                f"(similarity: {result['score']:.3f}) ---\n"
            )
            parts.append(pil_img)

            # Inject OCR text as ground truth — overrides Gemini's visual guesses
            ocr = result.get("ocr_text", "").strip()
            if ocr:
                parts.append(
                    f"\n[OCR TEXT FROM THIS PAGE — treat as ground truth, "
                    f"do not override with your own knowledge]:\n{ocr}\n[END OCR]\n\n"
                )

        parts.append(
            "\n\nAnswer the user's question based ONLY on what you can see in the "
            "image(s) above. Be specific, cite page numbers and exact labels."
        )

        response = model.generate_content(
            parts,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=settings.MAX_TOKENS_GENERATION,
                temperature=0.1,   # lower = less hallucination
            ),
        )

        return response.text
