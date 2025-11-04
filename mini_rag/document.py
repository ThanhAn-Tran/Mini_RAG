"""Document loading and preprocessing utilities."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_text_from_pdf(
    pdf_path: Path,
    *,
    use_ocr: bool = True,
    ocr_lang: str = "vie+eng",
    poppler_path: Optional[str] = None,
) -> str:
    """Extract text from a PDF, falling back to OCR when necessary."""
    logger.info("Loading document %s", pdf_path)
    text = ""

    try:
        import pdfplumber  # type: ignore
    except ImportError:
        logger.info("pdfplumber not installed; skipping embedded text extraction.")
    else:
        with pdfplumber.open(str(pdf_path)) as pdf:
            segments = []
            for page_idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                logger.debug("Page %d (pdfplumber) -> %d characters", page_idx, len(page_text))
                segments.append(page_text)
        text = "\n".join(segments)
        logger.info("pdfplumber extracted %d characters.", len(text))

    if text and len(text.strip()) > 150:
        return text

    if not use_ocr:
        logger.warning("Text layer appears empty and OCR disabled.")
        return text

    try:
        import pytesseract  # type: ignore
        from pytesseract import TesseractNotFoundError  # type: ignore
    except ImportError:
        logger.warning(
            "pytesseract not installed. Install Tesseract OCR or provide --text to skip PDF extraction."
        )
        return text

    try:
        from PIL import Image  # type: ignore
    except ImportError:
        logger.warning("Pillow missing. Install Pillow for OCR support.")
        return text

    # Prefer PyMuPDF, otherwise fall back to pdf2image.
    images: List["Image.Image"] = []
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None  # type: ignore

    if fitz is not None:
        try:
            doc = fitz.open(str(pdf_path))
        except RuntimeError as exc:
            logger.warning("PyMuPDF failed to open PDF (%s).", exc)
        else:
            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(dpi=300, alpha=False)
                images.append(Image.open(io.BytesIO(pix.tobytes("png"))))

    if not images:
        try:
            from pdf2image import convert_from_path  # type: ignore
            from pdf2image.exceptions import (  # type: ignore
                PDFInfoNotInstalledError,
                PDFPageCountError,
                PDFSyntaxError,
            )
        except ImportError:
            logger.warning(
                "Neither PyMuPDF nor pdf2image available. Install `pymupdf` (recommended) or configure Poppler."
            )
            return text

        conversion_kwargs: Dict[str, object] = {"dpi": 300}
        if poppler_path:
            conversion_kwargs["poppler_path"] = poppler_path

        try:
            images = convert_from_path(str(pdf_path), **conversion_kwargs)
        except PDFInfoNotInstalledError as exc:
            raise RuntimeError(
                "Poppler not found. Install Poppler or set POPPLER_PATH / --poppler-path, or install pymupdf."
            ) from exc
        except (PDFPageCountError, PDFSyntaxError, OSError) as exc:
            raise RuntimeError(f"Failed to rasterize PDF for OCR: {exc}") from exc

    if not images:
        raise RuntimeError("Unable to render PDF pages for OCR.")

    logger.info("Falling back to OCR. This may take a few minutes...")
    ocr_texts: List[str] = []
    for idx, image in enumerate(images, start=1):
        try:
            page_text = pytesseract.image_to_string(image, lang=ocr_lang)
        except TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract OCR executable not found. Install Tesseract or update PATH."
            ) from exc
        logger.debug("Page %d (OCR) -> %d characters", idx, len(page_text))
        ocr_texts.append(page_text)
    ocr_result = "\n".join(ocr_texts)
    logger.info("OCR produced %d characters.", len(ocr_result))
    if text.strip():
        return f"{text}\n{ocr_result}"
    return ocr_result


def chunk_text(
    text: str,
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
) -> List[str]:
    """Chunk text by paragraph blocks with overlap."""
    lines = [line.strip() for line in text.splitlines()]
    blocks: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()

    for line in lines:
        if not line:
            flush()
            continue
        current.append(line)
    flush()

    segments: List[str] = []
    for block in blocks:
        words = block.split()
        if len(words) <= chunk_size:
            segments.append(block)
            continue

        start = 0
        total = len(words)
        while start < total:
            end = min(total, start + chunk_size)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                segments.append(chunk)
            if end >= total:
                break
            start = max(0, end - chunk_overlap)

    return segments


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


__all__ = ["extract_text_from_pdf", "chunk_text", "l2_normalize"]
