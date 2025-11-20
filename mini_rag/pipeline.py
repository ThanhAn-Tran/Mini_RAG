"""Core retrieval-augmented generation pipeline."""
from __future__ import annotations

import logging
import shutil
import os
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .config import DataConfig, GenerationConfig, RetrievalConfig
from .document import load_and_split_document

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
        self.embedder = HuggingFaceEmbeddings(
            model_name=retrieval.embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )

        persist_directory = "./chroma_db"
        should_reindex = data.force_reindex

        if should_reindex and os.path.exists(persist_directory):
            logger.info("Force reindex requested. Removing existing database...")
            shutil.rmtree(persist_directory)

        # Initialize Chroma (loads existing if available, or prepares to create new)
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedder,
            collection_metadata={"hnsw:space": "cosine"}
        )

        # Check if we need to ingest data (if DB is empty)
        if self.db._collection.count() == 0:
            logger.info("Database is empty or reindex requested. Loading and indexing data...")
            self.chunks = self._load_and_chunk_corpus()
            if not self.chunks:
                raise ValueError("Chunking produced no passages. Adjust chunk size or provide richer text.")

            logger.info("Embedding %d chunks into ChromaDB...", len(self.chunks))
            
            documents = [
                Document(page_content=chunk, metadata={"index": i})
                for i, chunk in enumerate(self.chunks)
            ]
            
            self.db.add_documents(documents)
            logger.info("Embedding completed.")
        else:
            logger.info("Loaded existing database with %d documents.", self.db._collection.count())
            logger.info("Skipping re-indexing. Use --reindex to force update.")

        logger.info("Request timeout set to %.1f seconds", generation.request_timeout)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_and_chunk_corpus(self) -> List[str]:
        data_config = self.data_config
        retrieval_config = self.retrieval_config
        
        file_path = None
        if data_config.text_path is not None and data_config.text_path.exists():
            file_path = data_config.text_path
        elif data_config.pdf_path.exists():
            file_path = data_config.pdf_path
            
        if not file_path:
             raise FileNotFoundError("No valid text or PDF file found in configuration.")

        if file_path.is_dir():
            logger.info("Loading documents from directory: %s", file_path)
            all_chunks = []
            for file in file_path.iterdir():
                if file.is_file() and file.suffix.lower() in ['.txt', '.pdf']:
                    try:
                        chunks = load_and_split_document(
                            file,
                            chunk_size=retrieval_config.chunk_size,
                            chunk_overlap=retrieval_config.chunk_overlap
                        )
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.warning("Failed to load file %s: %s", file, e)
            return all_chunks
        else:
            return load_and_split_document(
                file_path,
                chunk_size=retrieval_config.chunk_size,
                chunk_overlap=retrieval_config.chunk_overlap
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, question: str, *, top_k: Optional[int] = None) -> List[ContextItem]:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")
        top_k_value = top_k or self.retrieval_config.top_k
        
        results = self.db.similarity_search_with_score(question, k=top_k_value)
        
        context_items = []
        for doc, score in results:
            idx = doc.metadata.get("index", -1)
            context_items.append((doc.page_content, float(score), int(idx)))
            
        return context_items

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
                "Bạn là trợ lý AI chuyên về Tư tưởng Hồ Chí Minh. "
                "Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp. "
                "Trả lời chi tiết, chính xác và hoàn toàn bằng tiếng Việt."
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
            "- Dựa vào ngữ cảnh trên để trả lời câu hỏi.\n"
            "- Nếu ngữ cảnh không có thông tin, hãy nói 'Không tìm thấy thông tin trong tài liệu'.\n"
            "- Trả lời hoàn toàn bằng tiếng Việt.\n"
            "- Trích dẫn (Đoạn #) cho mỗi ý."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def call_model(self, messages: List[Dict[str, str]]) -> str:
        # LM Studio / OpenAI compatible payload
        # Note: Some local servers map 'frequency_penalty' to repetition penalty.
        # We will send both standard OpenAI params and some common local LLM params just in case.
        payload = {
            "model": self.generation_config.model_id,
            "messages": messages,
            "temperature": self.generation_config.temperature,
            "max_tokens": self.generation_config.max_output_tokens,
            "frequency_penalty": 0.5,  # Standard OpenAI param to reduce repetition
            "presence_penalty": 0.5,   # Standard OpenAI param to encourage new topics
            "repeat_penalty": self.generation_config.repetition_penalty, # Common local LLM param
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
