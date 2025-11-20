"""Command-line entrypoint for the Mini RAG demo."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

from typing import TYPE_CHECKING

from .config import AppConfig, DataConfig, GenerationConfig, RetrievalConfig

if TYPE_CHECKING:
    from .pipeline import RAGPipeline
    from .ui import build_demo


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini RAG demo with Gradio interface.")
    default_pdf = Path("data/gt-tu-tuong-ho-chi-minh-bo-gddt-2010.pdf")
    parser.add_argument("--pdf", type=Path, default=default_pdf, help="Đường dẫn tới file PDF nguồn.")
    parser.add_argument(
        "--text",
        type=Path,
        help="Đường dẫn tới file văn bản đã trích xuất (bỏ qua bước xử lý PDF).",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("LMSTUDIO_API_BASE", "http://127.0.0.1:1234"),
        help="Địa chỉ gốc của LM Studio/OpenAI-compatible API.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LMSTUDIO_MODEL", "vinallama-7b-chat"),
        help="Tên model LM Studio expose.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Tên embedding model từ sentence-transformers.",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Số lượng từ tối đa mỗi đoạn.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Số từ trùng giữa các đoạn.")
    parser.add_argument("--top-k", type=int, default=5, help="Số đoạn dùng để trả lời.")
    parser.add_argument("--no-ocr", action="store_true", help="Tắt OCR fallback nếu PDF đã có text.")
    parser.add_argument(
        "--ocr-lang",
        default=os.environ.get("OCR_LANG", "vie+eng"),
        help="Ngôn ngữ OCR cho Tesseract (ví dụ: vie+eng).",
    )
    parser.add_argument("--reindex", action="store_true", help="Buộc xóa database cũ và index lại dữ liệu.")
    parser.add_argument(
        "--poppler-path",
        default=os.environ.get("POPPLER_PATH"),
        help="Đường dẫn tới thư mục chứa binary Poppler (Windows).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1200.0,
        help="Timeout (giây) cho request tới LM Studio (mặc định 1200).",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature cho model sinh.")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Giới hạn token output gửi tới model.",
    )
    parser.add_argument(
        "--system-prompt",
        help="Ghi đè system prompt (chuỗi). Nếu muốn đọc từ file, dùng --system-prompt-file.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Đường dẫn tới file chứa system prompt tùy chỉnh.",
    )
    parser.add_argument("--share", action="store_true", help="Bật share link của Gradio.")
    parser.add_argument("--port", type=int, default=7860, help="Cổng chạy Gradio.")
    parser.add_argument(
        "--examples",
        nargs="*",
        help="Danh sách câu hỏi mẫu hiển thị trong giao diện.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> AppConfig:
    if args.system_prompt and args.system_prompt_file:
        raise ValueError("Chỉ chọn một trong --system-prompt hoặc --system-prompt-file.")
    system_prompt: Optional[str] = None
    if args.system_prompt_file:
        if not args.system_prompt_file.exists():
            raise FileNotFoundError(f"Không tìm thấy system prompt file: {args.system_prompt_file}")
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
    elif args.system_prompt:
        system_prompt = args.system_prompt

    data_cfg = DataConfig(
        pdf_path=args.pdf,
        text_path=args.text,
        use_ocr=not args.no_ocr,
        ocr_lang=args.ocr_lang,
        poppler_path=args.poppler_path,
        force_reindex=args.reindex,
    )
    retrieval_cfg = RetrievalConfig(
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )
    generation_cfg = GenerationConfig(
        api_base=args.api_base,
        model_id=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        request_timeout=args.request_timeout,
        system_prompt=system_prompt,
    )

    examples: List[str]
    if args.examples:
        examples = list(args.examples)
    else:
        examples = [
            "Trình bày quan điểm của Hồ Chí Minh về độc lập dân tộc?",
            "Phân tích nội dung cốt lõi của tư tưởng Hồ Chí Minh.",
            "Tư tưởng Hồ Chí Minh về xây dựng Đảng cầm quyền gồm những điểm nào?",
        ]

    return AppConfig(
        data=data_cfg,
        retrieval=retrieval_cfg,
        generation=generation_cfg,
        examples=examples,
        port=args.port,
        share=args.share,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = build_config(args)
    if not config.data.pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp PDF: {config.data.pdf_path}")
    if config.data.text_path is not None and not config.data.text_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp văn bản: {config.data.text_path}")

    from .pipeline import RAGPipeline  # pylint: disable=import-outside-toplevel
    from .ui import build_demo  # pylint: disable=import-outside-toplevel

    pipeline = RAGPipeline(
        data=config.data,
        retrieval=config.retrieval,
        generation=config.generation,
    )
    demo = build_demo(pipeline, config.examples)
    demo.launch(server_port=config.port, share=config.share)


if __name__ == "__main__":
    main()
