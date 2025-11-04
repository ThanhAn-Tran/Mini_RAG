"""Configuration dataclasses for the Mini RAG application."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    pdf_path: Path
    text_path: Optional[Path] = None
    use_ocr: bool = True
    ocr_lang: str = "vie+eng"
    poppler_path: Optional[str] = None


@dataclass
class RetrievalConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 200
    chunk_overlap: int = 50
    top_k: int = 5


@dataclass
class GenerationConfig:
    api_base: str = "http://127.0.0.1:1234"
    model_id: str = "llama-2-7b-chat"
    temperature: float = 0.1
    max_output_tokens: int = 512
    request_timeout: float = 1200.0
    system_prompt: Optional[str] = None


@dataclass
class AppConfig:
    data: DataConfig
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    examples: List[str] = field(default_factory=list)
    port: int = 7860
    share: bool = False


__all__ = [
    "DataConfig",
    "RetrievalConfig",
    "GenerationConfig",
    "AppConfig",
]
