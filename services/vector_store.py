import os
import pickle
from typing import Optional

import faiss
import numpy as np

from config import settings
from services.pdf_processor import PageTile


class VectorStore:
    def __init__(self, dim: int = settings.EMBEDDING_DIM):
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._tiles: list[PageTile] = []

    def add(self, vectors: np.ndarray, tiles: list[PageTile]):
        assert vectors.shape[0] == len(tiles)
        assert vectors.shape[1] == self.dim
        self._index.add(vectors.astype(np.float32))
        self._tiles.extend(tiles)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        document_id: Optional[str] = None,
    ) -> list[dict]:
        if self._index.ntotal == 0:
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        fetch_k = min(top_k * 5 if document_id else top_k, self._index.ntotal)

        scores, indices = self._index.search(query, fetch_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            tile = self._tiles[idx]
            if document_id and tile.doc_id != document_id:
                continue

            results.append({
                "doc_id": tile.doc_id,
                "page": tile.page_number,
                "score": float(score),
                "image_b64": tile.image_b64,
                "hires_b64": tile.hires_b64,
                "ocr_text": tile.ocr_text,     
                "metadata": tile.metadata,
            })

            if len(results) >= top_k:
                break

        return results

    def total_vectors(self) -> int:
        return self._index.ntotal

    def list_documents(self) -> list[str]:
        seen = []
        for tile in self._tiles:
            if tile.doc_id not in seen:
                seen.append(tile.doc_id)
        return seen

    def clear(self):
        self._index.reset()
        self._tiles.clear()

    def save(self, path: str = "faiss_index"):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "tiles.pkl"), "wb") as f:
            pickle.dump(self._tiles, f)

    def load(self, path: str = "faiss_index"):
        self._index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "tiles.pkl"), "rb") as f:
            self._tiles = pickle.load(f)
