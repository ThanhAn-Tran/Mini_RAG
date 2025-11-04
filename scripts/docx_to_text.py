#!/usr/bin/env python
"""Quick DOCX to text converter to support the Mini RAG pipeline."""
from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

try:
    import docx
except ImportError as exc:
    docx = None  # type: ignore


LOGGER = logging.getLogger("docx_to_text")


def extract_text(docx_path: Path) -> str:
    if docx is not None:
        document = docx.Document(str(docx_path))
        parts = []
        for para in document.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)
        return "\n".join(parts)

    LOGGER.info("python-docx not available; falling back to raw XML extraction.")
    with zipfile.ZipFile(docx_path) as docx_zip:
        try:
            xml_bytes = docx_zip.read("word/document.xml")
        except KeyError as exc:
            raise RuntimeError("Word document is missing word/document.xml") from exc

    import xml.etree.ElementTree as ET

    tree = ET.fromstring(xml_bytes)
    namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts = []

    for paragraph in tree.findall(".//w:body/w:p", namespaces):
        texts = [
            node.text
            for node in paragraph.findall(".//w:t", namespaces)
            if node.text
        ]
        para_text = "".join(texts).strip()
        if para_text:
            parts.append(para_text)
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DOCX file to plain text.")
    parser.add_argument("--docx", type=Path, required=True, help="Đường dẫn tới file .docx nguồn.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Đường dẫn file .txt để lưu kết quả. Mặc định trùng tên.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    docx_path: Path = args.docx
    if not docx_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file DOCX: {docx_path}")

    LOGGER.info("Reading DOCX from %s", docx_path)
    text = extract_text(docx_path)
    if not text.strip():
        LOGGER.warning("Không trích xuất được nội dung. DOCX có thể rỗng hoặc chứa toàn hình ảnh.")

    output_path = args.output or docx_path.with_suffix(".txt")
    output_path.write_text(text, encoding="utf-8")
    LOGGER.info("Đã lưu văn bản vào %s", output_path)


if __name__ == "__main__":
    main()
