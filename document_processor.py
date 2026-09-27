"""
document_processor.py
----------------------
Utilities to extract raw text from uploaded documents (PDF, DOCX, TXT)
and split that text into overlapping chunks suitable for retrieval.
"""

from typing import List
import io

import pypdf
import docx


def extract_text(uploaded_file) -> str:
    """
    Extract raw text from a Streamlit UploadedFile object.
    Supports .pdf, .docx, and .txt files.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return _extract_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        return _extract_docx(uploaded_file)
    elif filename.endswith(".txt"):
        return _extract_txt(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def _extract_pdf(uploaded_file) -> str:
    reader = pypdf.PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_docx(uploaded_file) -> str:
    document = docx.Document(io.BytesIO(uploaded_file.read()))
    return "\n".join(p.text for p in document.paragraphs)


def _extract_txt(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text: full document text
        chunk_size: approximate number of words per chunk
        overlap: number of words to overlap between consecutive chunks

    Returns:
        List of text chunks (empty/whitespace-only chunks are dropped).
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks
