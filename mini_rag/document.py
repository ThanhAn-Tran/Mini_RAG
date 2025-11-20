"""Document loading and preprocessing utilities using LangChain."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_and_split_document(
    file_path: Path,
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
) -> List[str]:
    """Load document and split into chunks using LangChain."""
    logger.info("Loading document %s", file_path)
    
    if file_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
        loader = TextLoader(str(file_path), encoding="utf-8")
        
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    split_docs = text_splitter.split_documents(documents)
    logger.info("Split into %d chunks.", len(split_docs))
    
    return [doc.page_content for doc in split_docs]


__all__ = ["load_and_split_document"]
