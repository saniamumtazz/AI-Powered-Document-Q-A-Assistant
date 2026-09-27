"""
qa_engine.py
------------
Retrieval-augmented Q&A engine.

Retrieval: TF-IDF vectorization + cosine similarity over document chunks
           (no external vector DB needed — lightweight and dependency-free).
Generation: sends the top-k retrieved chunks + the user's question to an
            LLM (Google Gemini or OpenAI) with a prompt that instructs it
            to answer only from the provided context.
"""

from typing import List, Tuple
import os

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided document context. If the answer is not contained in the "
    "context, say so honestly instead of guessing. Be concise and clear."
)


class DocumentQAEngine:
    def __init__(self, provider: str = "gemini", api_key: str = ""):
        self.provider = provider
        self.api_key = api_key
        self.chunks: List[str] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = None

    def build_index(self, chunks: List[str]) -> None:
        """Fit the TF-IDF vectorizer over the document chunks."""
        self.chunks = chunks
        self.doc_matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, question: str, top_k: int = 3) -> List[str]:
        """Return the top_k most relevant chunks for the question."""
        if self.doc_matrix is None or not self.chunks:
            return []

        query_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0] or [
            self.chunks[i] for i in top_indices
        ]

    def answer_question(self, question: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """
        Retrieve relevant chunks, then call the configured LLM to
        generate a grounded answer. Returns (answer, source_chunks).
        """
        relevant_chunks = self.retrieve(question, top_k=top_k)
        context = "\n\n---\n\n".join(relevant_chunks)

        prompt = (
            f"Document context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer the question using only the context above."
        )

        if self.provider == "gemini":
            answer = self._call_gemini(prompt)
        elif self.provider == "openai":
            answer = self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return answer, relevant_chunks

    def _call_gemini(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("Missing Gemini API key.")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]}
            ]
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_openai(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("Missing OpenAI API key.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
