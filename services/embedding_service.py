import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import asyncio
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import settings
from services.pdf_processor import PageTile

if TYPE_CHECKING:
    from services.vector_store import VectorStore


class EmbeddingService:
    def __init__(self, model_name: str = settings.CLIP_MODEL_NAME):
        self.model_name = model_name
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        if self._model is None:
            print(f"[EmbeddingService] Loading CLIP model: {self.model_name}")
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            print(f"[EmbeddingService] Model loaded on {self.device}")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Return a normalised float32 embedding (dim=512) for a PIL Image."""
        self._load()
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)

        vec = features.squeeze().cpu().numpy().astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)   # L2-normalise

    def embed_images_batch(
        self, images: list[Image.Image], batch_size: int = 8
    ) -> np.ndarray:
        self._load()
        all_vecs: list[np.ndarray] = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self._processor(images=batch, return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                features = self._model.get_image_features(**inputs)

            vecs = features.cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            all_vecs.append(vecs / norms)

        return np.vstack(all_vecs)

    def embed_text(self, text: str) -> np.ndarray:
        self._load()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(
            self.device
        )

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)

        vec = features.squeeze().cpu().numpy().astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)

    async def embed_and_index(
        self, pages: list[PageTile], vector_store: "VectorStore"
    ) -> int:
        loop = asyncio.get_event_loop()

        def _blocking():
            images = [p.image for p in pages]
            vectors = self.embed_images_batch(images)
            vector_store.add(vectors, pages)
            return len(pages)

        count = await loop.run_in_executor(None, _blocking)
        return count
