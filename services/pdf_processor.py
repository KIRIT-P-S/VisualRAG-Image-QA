"""
PDF Processor
-------------
Stores TWO versions of each page:
  - embed_image  : small (512×512) for CLIP embedding — fast, low memory
  - hires_image  : high-res (1600×1600) for Gemini Vision — readable text
  - hires_b64    : base64 of high-res image sent to Gemini
  - image_b64    : base64 of small image shown as thumbnail in frontend
"""

import io
import base64
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from pdf2image import convert_from_bytes

from config import settings


@dataclass
class PageTile:
    """Represents a single rendered PDF page with dual-resolution images."""
    # ── Required fields (no defaults) — must come first ───────────────────────
    doc_id: str
    page_number: int
    image: Image.Image          # Small image for CLIP embedding

    # ── Optional fields (with defaults) — must come after required fields ──────
    hires_image: Image.Image = field(default=None, repr=False)   # High-res for Gemini
    image_b64: str = field(default="", repr=False)               # Small thumbnail → frontend
    hires_b64: str = field(default="", repr=False)               # High-res → Gemini Vision
    ocr_text: str = field(default="", repr=False)                # OCR text → ground truth
    metadata: dict = field(default_factory=dict)


class PDFProcessor:
    """
    Converts a PDF byte stream into PageTile objects with dual-resolution images.

    Flow:
        PDF bytes
          ├─ render at HIGH DPI (300) → hires_image (for Gemini)
          └─ downscale to 512×512    → embed_image (for CLIP)
    """

    def pdf_to_images(
        self,
        pdf_bytes: bytes,
        doc_id: str,
        embed_dpi: int = 150,
        hires_dpi: int = 300,
        embed_max: tuple = (512, 512),
        hires_max: tuple = (1600, 1600),
    ) -> list[PageTile]:
        """
        Render every page at two resolutions.

        Args:
            pdf_bytes : Raw PDF bytes.
            doc_id    : Unique document identifier.
            embed_dpi : DPI for CLIP embedding images (small, fast).
            hires_dpi : DPI for Gemini Vision images (large, readable).
            embed_max : Max pixel size for embedding images.
            hires_max : Max pixel size for Gemini images.

        Returns:
            List of PageTile, one per page.
        """
        # ── Render small images for CLIP ──────────────────────────────────────
        small_pages: list[Image.Image] = convert_from_bytes(
            pdf_bytes, dpi=embed_dpi, fmt="RGB"
        )

        # ── Render high-res images for Gemini ────────────────────────────────
        hires_pages: list[Image.Image] = convert_from_bytes(
            pdf_bytes, dpi=hires_dpi, fmt="RGB"
        )

        tiles: list[PageTile] = []
        for page_num, (small, hires) in enumerate(
            zip(small_pages, hires_pages), start=1
        ):
            # Resize small image for CLIP
            small = small.convert("RGB")
            small.thumbnail(embed_max, Image.LANCZOS)

            # Resize high-res image (cap at hires_max to avoid OOM)
            hires = hires.convert("RGB")
            hires.thumbnail(hires_max, Image.LANCZOS)

            # Extract OCR text from high-res image (best effort — skipped if tesseract missing)
            ocr_text = self._extract_ocr(hires)

            tile = PageTile(
                doc_id=doc_id,
                page_number=page_num,
                image=small,
                hires_image=hires,
                image_b64=self._to_base64(small, quality=80),
                hires_b64=self._to_base64(hires, quality=92),
                ocr_text=ocr_text,
                metadata={
                    "doc_id": doc_id,
                    "page": page_num,
                    "embed_size": (small.width, small.height),
                    "hires_size": (hires.width, hires.height),
                    "ocr_chars": len(ocr_text),
                },
            )
            tiles.append(tile)

        return tiles

    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _to_base64(image: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
        buf = io.BytesIO()
        image.save(buf, format=fmt, quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _extract_ocr(image: Image.Image) -> str:
        """
        Extract text using Tesseract OCR.
        Returns empty string silently if tesseract is not installed.

        Install tesseract:
          Windows: https://github.com/UB-Mannheim/tesseract/wiki
          Ubuntu:  sudo apt install tesseract-ocr
          macOS:   brew install tesseract
        """
        try:
            import pytesseract

            # Windows default install path — adjust if yours is different
            import platform
            if platform.system() == "Windows":
                pytesseract.pytesseract.tesseract_cmd = (
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                )

            # 2x upscale improves OCR accuracy on small schematic text
            w, h = image.size
            big = image.resize((w * 2, h * 2), Image.LANCZOS)

            # psm 11 = sparse text (good for schematics with labels scattered around)
            text = pytesseract.image_to_string(big, config="--psm 11")
            return text.strip()

        except ImportError:
            # pytesseract not installed — OCR disabled, Gemini uses image only
            return ""
        except Exception:
            # tesseract binary not found or other error — fail silently
            return ""
