"""Core retrieval-augmented generation pipeline."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from .config import DataConfig, GenerationConfig, RetrievalConfig
from .document import chunk_text, extract_text_from_pdf, l2_normalize

logger = logging.getLogger(__name__)

ContextItem = Tuple[str, float, int]


class RAGPipeline:
    """End-to-end pipeline managing embeddings, retrieval, and generation."""

    def __init__(
        self,
        *,
        data: DataConfig,
        retrieval: RetrievalConfig,
        generation: GenerationConfig,
    ) -> None:
        self.data_config = data
        self.retrieval_config = retrieval
        self.generation_config = generation
        self.system_prompt = generation.system_prompt

        logger.info("Loading embedding model: %s", retrieval.embedding_model)
        self.embedder = SentenceTransformer(retrieval.embedding_model)

        raw_text = self._load_corpus()
        if not raw_text.strip():
            raise ValueError(
                "No text found in the provided sources. Provide --text or install OCR dependencies."
            )

        self.chunks = chunk_text(
            raw_text,
            chunk_size=retrieval.chunk_size,
            chunk_overlap=retrieval.chunk_overlap,
        )
        if not self.chunks:
            raise ValueError("Chunking produced no passages. Adjust chunk size or provide richer text.")

        logger.info("Embedding %d chunks...", len(self.chunks))
        embeddings = self.embedder.encode(
            self.chunks,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.embeddings = l2_normalize(np.array(embeddings, dtype=np.float32))
        logger.info("Embedding completed.")
        logger.info("Request timeout set to %.1f seconds", generation.request_timeout)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_corpus(self) -> str:
        data_config = self.data_config
        if data_config.text_path is not None:
            logger.info("Loading pre-extracted text from %s", data_config.text_path)
            if not data_config.text_path.exists():
                raise FileNotFoundError(f"Text file not found: {data_config.text_path}")
            return data_config.text_path.read_text(encoding="utf-8")

        if not data_config.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {data_config.pdf_path}")

        return extract_text_from_pdf(
            pdf_path=data_config.pdf_path,
            use_ocr=data_config.use_ocr,
            ocr_lang=data_config.ocr_lang,
            poppler_path=data_config.poppler_path,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, question: str, *, top_k: Optional[int] = None) -> List[ContextItem]:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")
        top_k_value = top_k or self.retrieval_config.top_k
        query_embedding = self.embedder.encode([
            question
        ], show_progress_bar=False, convert_to_numpy=True)
        query_embedding = l2_normalize(np.array(query_embedding, dtype=np.float32))[0]
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k_value]
        return [(self.chunks[idx], float(scores[idx]), int(idx)) for idx in top_indices]

    # ------------------------------------------------------------------
    # Prompting helpers
    # ------------------------------------------------------------------
    def build_messages(
        self,
        question: str,
        contexts: Sequence[ContextItem],
        *,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        merged_prompt = system_prompt or self.system_prompt
        if merged_prompt is None:
            system_prompt = (
                "Bạn là học giả trợ lý chuyên phân tích giáo trình tư tưởng Hồ Chí Minh. "
                "Luôn trả lời bằng tiếng Việt. Chỉ sử dụng thông tin trong các đoạn ngữ cảnh đã cho, trích dẫn rõ."
            )
        else:
            system_prompt = merged_prompt

        context_blocks: List[str] = []
        for _, (chunk, score, idx) in enumerate(contexts, start=1):
            context_blocks.append(f"[Đoạn {idx + 1} · điểm {score:.3f}]\n{chunk}")
        context_text = "\n\n".join(context_blocks) if context_blocks else "Không có ngữ cảnh."

        user_prompt = (
            f"Câu hỏi: {question}\n\n"
            f"Ngữ cảnh:\n{context_text}\n\n"
            "Hướng dẫn:\n"
            "- Nếu ngữ cảnh liên quan, trả lời rõ ràng, chia thành các mục nếu phù hợp.\n"
            "- Chèn trích dẫn dạng (Đoạn #) ngay sau luận điểm lấy từ ngữ cảnh.\n"
            "- Nếu ngữ cảnh chưa đủ, nói rõ 'Chưa có thông tin đủ để kết luận từ tài liệu'."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def call_model(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.generation_config.model_id,
            "messages": messages,
            "temperature": self.generation_config.temperature,
            "max_tokens": self.generation_config.max_output_tokens,
        }
        url = f"{self.generation_config.api_base.rstrip('/')}/v1/chat/completions"
        response = requests.post(url, json=payload, timeout=self.generation_config.request_timeout)
        if response.status_code == 404:
            logger.warning("/v1/chat/completions not found; falling back to /v1/completions")
            return self._call_completion(messages)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"No choices returned: {data}")
        message = choices[0].get("message", {})
        return message.get("content", "").strip()

    def _call_completion(self, messages: List[Dict[str, str]]) -> str:
        prompt_parts: List[str] = []
        for message in messages:
            role = message["role"].capitalize()
            prompt_parts.append(f"{role}: {message['content']}")
        prompt_parts.append("Assistant:")
        payload = {
            "model": self.generation_config.model_id,
            "prompt": "\n\n".join(prompt_parts),
            "temperature": self.generation_config.temperature,
            "max_tokens": self.generation_config.max_output_tokens,
        }
        url = f"{self.generation_config.api_base.rstrip('/')}/v1/completions"
        response = requests.post(url, json=payload, timeout=self.generation_config.request_timeout)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"No choices returned: {data}")
        return choices[0].get("text", "").strip()

    # ------------------------------------------------------------------
    def answer(self, question: str, *, top_k: Optional[int] = None) -> Dict[str, object]:
        contexts = self.retrieve(question, top_k=top_k)
        messages = self.build_messages(question, contexts)
        answer = self.call_model(messages)
        return {"answer": answer, "contexts": contexts}

    @staticmethod
    def contexts_to_markdown(contexts: Sequence[ContextItem]) -> str:
        if not contexts:
            return "Không tìm thấy đoạn văn liên quan."
        lines: List[str] = []
        for _, (chunk, score, idx) in enumerate(contexts, start=1):
            lines.append(f"**Đoạn {idx + 1} · điểm {score:.3f}**")
            lines.append(chunk)
            lines.append("")
        return "\n".join(lines).strip()


__all__ = ["RAGPipeline", "ContextItem"]
