"""
AI-Powered Document Q&A Assistant
----------------------------------
Streamlit app that lets a user upload a document (PDF / DOCX / TXT),
then ask natural-language questions about it. The app retrieves the
most relevant chunks of the document using TF-IDF + cosine similarity,
and then sends those chunks + the question to an LLM API (Google Gemini
by default, OpenAI supported as an alternative) to generate a grounded
answer.

Run locally:
    streamlit run app.py

Author: Sania Mumtaz
"""

import os
import streamlit as st
from dotenv import load_dotenv

from document_processor import extract_text, chunk_text
from qa_engine import DocumentQAEngine

load_dotenv()

st.set_page_config(
    page_title="AI Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
)


def init_session_state():
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "doc_name" not in st.session_state:
        st.session_state.doc_name = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def sidebar_config():
    st.sidebar.title("⚙️ Configuration")

    provider = st.sidebar.selectbox(
        "LLM Provider",
        options=["gemini", "openai"],
        help="Choose which LLM API to use for generating answers.",
    )

    env_key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    default_key = os.getenv(env_key_name, "")

    api_key = st.sidebar.text_input(
        f"{provider.title()} API Key",
        value=default_key,
        type="password",
        help="You can also set this via a .env file (see README).",
    )

    top_k = st.sidebar.slider(
        "Relevant chunks to retrieve (top_k)",
        min_value=1,
        max_value=8,
        value=3,
        help="How many document chunks to feed the LLM as context.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This app performs retrieval-augmented Q&A: it finds the most "
        "relevant parts of your document with TF-IDF similarity search, "
        "then asks the LLM to answer using only that context."
    )

    return provider, api_key, top_k


def main():
    init_session_state()
    st.title("📄 AI-Powered Document Q&A Assistant")
    st.caption("Upload a document and ask questions about it in plain English.")

    provider, api_key, top_k = sidebar_config()

    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file is not None:
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner("Reading and indexing document..."):
                raw_text = extract_text(uploaded_file)
                chunks = chunk_text(raw_text, chunk_size=500, overlap=50)

                if not chunks:
                    st.error("Could not extract any readable text from this file.")
                    return

                engine = DocumentQAEngine(provider=provider, api_key=api_key)
                engine.build_index(chunks)

                st.session_state.engine = engine
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []

            st.success(f"Indexed **{uploaded_file.name}** — {len(chunks)} chunks ready.")
        else:
            # Same document already indexed; just refresh provider/key if changed
            st.session_state.engine.provider = provider
            st.session_state.engine.api_key = api_key

    if st.session_state.engine is None:
        st.info("Upload a document above to get started.")
        return

    st.markdown("### 💬 Ask a question")
    question = st.text_input("Your question", placeholder="e.g. What is the main conclusion of this document?")
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and question.strip():
        if not api_key:
            st.error(f"Please provide a {provider.title()} API key in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer, sources = st.session_state.engine.answer_question(question, top_k=top_k)
                    st.session_state.chat_history.append(
                        {"question": question, "answer": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"Error while generating answer: {e}")

    # Display chat history (most recent first)
    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            with st.expander("📚 Source chunks used"):
                for i, src in enumerate(turn["sources"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(src[:500] + ("..." if len(src) > 500 else ""))


if __name__ == "__main__":
    main()
