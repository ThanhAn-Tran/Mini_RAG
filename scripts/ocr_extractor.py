#!/usr/bin/env python
"""Utility script to OCR a PDF into plain text using PyMuPDF and EasyOCR."""
from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    import fitz  # type: ignore
except ImportError as exc:
    raise ImportError(
        "PyMuPDF (pip install pymupdf) is required for this OCR helper."
    ) from exc

try:
    import easyocr  # type: ignore
except ImportError as exc:
    raise ImportError(
        "EasyOCR (pip install easyocr) is required for this OCR helper."
    ) from exc

try:
    from PIL import Image  # type: ignore
except ImportError as exc:
    raise ImportError("Pillow is required for image handling. Install it with `pip install Pillow`.") from exc


LOGGER = logging.getLogger("ocr_extractor")


def pdf_to_images(pdf_path: Path, dpi: int) -> Iterable[Tuple[int, Image.Image]]:
    doc = fitz.open(str(pdf_path))
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        yield page_index, image


def run_easyocr(
    pdf_path: Path,
    languages: Sequence[str],
    dpi: int = 250,
    gpu: bool | None = None,
    paragraph: bool = True,
) -> List[str]:
    if gpu is None:
        gpu = easyocr.utils.is_gpu_available()
    LOGGER.info("Initialising EasyOCR with languages=%s | gpu=%s", languages, gpu)
    reader = easyocr.Reader(list(languages), gpu=gpu)

    collected: List[str] = []
    for page_index, pil_image in pdf_to_images(pdf_path, dpi=dpi):
        LOGGER.info("Processing page %d", page_index + 1)
        np_image = np.array(pil_image.convert("RGB"))
        results = reader.readtext(np_image, detail=0, paragraph=paragraph)
        page_text = "\n".join(results).strip()
        collected.append(page_text)

    return collected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR a PDF to plain text using EasyOCR.")
    parser.add_argument("--pdf", type=Path, required=True, help="Đường dẫn tới PDF nguồn (ảnh).")
    parser.add_argument(
        "--output",
        type=Path,
        help="File .txt để lưu kết quả. Mặc định trùng tên PDF.",
    )
    parser.add_argument(
        "--lang",
        nargs="*",
        default=["vi", "en"],
        help="Danh sách mã ngôn ngữ EasyOCR (ví dụ vi en).",
    )
    parser.add_argument("--dpi", type=int, default=250, help="DPI để render trang PDF.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Buộc EasyOCR chạy trên CPU (mặc định tự phát hiện GPU).",
    )
    parser.add_argument(
        "--no-paragraph",
        action="store_true",
        help="Không gom thành đoạn văn (EasyOCR detail=0, paragraph=False).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy PDF: {pdf_path}")

    output_path = args.output or pdf_path.with_suffix(".txt")

    gpu_flag = False if args.cpu else None

    texts = run_easyocr(
        pdf_path=pdf_path,
        languages=args.lang,
        dpi=args.dpi,
        gpu=gpu_flag,
        paragraph=not args.no_paragraph,
    )

    merged = []
    for idx, page_text in enumerate(texts, start=1):
        merged.append(f"[Trang {idx}]")
        merged.append(page_text)
        merged.append("")

    output_path.write_text("\n".join(merged).strip(), encoding="utf-8")
    LOGGER.info("📄 Đã lưu văn bản OCR vào %s", output_path)


if __name__ == "__main__":
    main()
