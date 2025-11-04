"""Gradio interface for the Mini RAG demo."""
from __future__ import annotations

from typing import List, Optional, Tuple

import gradio as gr  # type: ignore

from .pipeline import RAGPipeline


def build_demo(rag: RAGPipeline, examples: Optional[List[str]] = None):
    """Return a Gradio Blocks application bound to the provided pipeline."""

    def respond(question: str) -> Tuple[str, str]:
        if not question or not question.strip():
            return "Vui lòng nhập câu hỏi.", ""
        try:
            result = rag.answer(question)
        except Exception as exc:  # pylint: disable=broad-except
            return f"Lỗi: {exc}", ""
        answer = result["answer"]
        context_md = RAGPipeline.contexts_to_markdown(result["contexts"])
        return answer, context_md

    with gr.Blocks(title="Mini RAG demo") as demo:
        gr.Markdown("# Mini RAG demo\nSử dụng LM Studio (llama-2-7b-chat) để sinh câu trả lời.")
        source_name = (
            rag.data_config.text_path.name
            if rag.data_config.text_path is not None
            else rag.data_config.pdf_path.name
        )
        gr.Markdown(
            f"Nguồn: `{source_name}` · Tổng đoạn: {len(rag.chunks)} · Top-k: {rag.retrieval_config.top_k}"
        )
        question = gr.Textbox(
            label="Câu hỏi",
            placeholder="Ví dụ: Trình bày quan điểm của Hồ Chí Minh về độc lập dân tộc?",
            lines=2,
        )
        answer = gr.Markdown(label="Trả lời")
        context = gr.Markdown(label="Đoạn tham chiếu")
        ask_btn = gr.Button("Hỏi")
        ask_btn.click(respond, inputs=[question], outputs=[answer, context])

        if examples:
            gr.Examples(examples=examples, inputs=[question])

    return demo


__all__ = ["build_demo"]
