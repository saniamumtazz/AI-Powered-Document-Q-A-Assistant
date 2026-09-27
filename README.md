# 📄 AI-Powered Document Q&A Assistant

A Python application that lets you upload a document (PDF, DOCX, or TXT)
and ask natural-language questions about it. The app retrieves the most
relevant sections of the document and uses an LLM to generate an accurate,
grounded answer — a lightweight Retrieval-Augmented Generation (RAG) pipeline.

## ✨ Features

- 📤 Upload PDF, DOCX, or TXT files
- ✂️ Automatic text extraction and chunking
- 🔍 TF-IDF + cosine similarity retrieval (no vector database required)
- 🤖 Pluggable LLM backend — Google Gemini or OpenAI
- 💬 Chat-style interface with source-chunk transparency
- ⚙️ Configurable number of retrieved chunks (top-k)

## 🧱 Tech Stack

- **Language:** Python
- **UI:** Streamlit
- **Retrieval:** scikit-learn (TF-IDF, cosine similarity)
- **Document parsing:** pypdf, python-docx
- **LLM APIs:** Google Gemini API / OpenAI API
- **Prompt Engineering:** context-grounded system prompt to reduce hallucination

## 📂 Project Structure

```
doc-qa-assistant/
├── app.py                  # Streamlit UI and app flow
├── document_processor.py   # Text extraction + chunking
├── qa_engine.py             # TF-IDF retrieval + LLM API calls
├── requirements.txt
├── .env.example
├── sample_docs/
│   └── sample.txt
└── README.md
```

## 🚀 Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/saniamumtazz/doc-qa-assistant.git
   cd doc-qa-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API key**
   ```bash
   cp .env.example .env
   # then open .env and paste your GEMINI_API_KEY or OPENAI_API_KEY
   ```
   Get a free Gemini key at https://aistudio.google.com/app/apikey

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the local URL Streamlit prints (usually `http://localhost:8501`),
   upload a document (try `sample_docs/sample.txt`), and start asking questions!

## 🛠 How It Works

1. **Indexing** — the uploaded document is parsed into raw text, then split
   into overlapping ~500-word chunks.
2. **Retrieval** — when you ask a question, it's compared against every
   chunk using TF-IDF vectors and cosine similarity; the top-k most similar
   chunks are selected.
3. **Generation** — those chunks are inserted into a prompt along with your
   question and sent to the chosen LLM, which is instructed to answer only
   from the given context (reducing hallucination).

## 🔮 Possible Extensions

- Swap TF-IDF retrieval for dense embeddings (e.g. `sentence-transformers`)
- Add a persistent vector store (FAISS / ChromaDB) for large document sets
- Support multi-document / multi-file Q&A
- Add answer citations linking back to exact page numbers

## 📄 License

MIT
