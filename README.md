# pdf.qa — Local PDF Q&A RAG App

A fully local, privacy-friendly PDF question-answering app built with **FastAPI**, **FAISS**, **Sentence Transformers**, and **Ollama**. Upload any PDF and ask questions about it in plain English — no cloud APIs, no data leaving your machine.

---

## What It Does

You upload a PDF. The app reads it, splits it into chunks, and converts those chunks into semantic vectors stored in a FAISS index. When you ask a question, your question is also converted into a vector and compared against the index to find the most relevant chunks. Those chunks are then passed to a local LLM (tinyllama via Ollama) which generates a natural language answer.

This pattern is called **RAG — Retrieval Augmented Generation**.

---

## How It Works (Architecture)

```
PDF Upload
    │
    ▼
Extract Text (PyMuPDF)
    │
    ▼
Split into Chunks (800 chars, 100 overlap)
    │
    ▼
Embed Chunks → Vectors (all-MiniLM-L6-v2)
    │
    ▼
Store in FAISS Index (in memory)
    
User Question
    │
    ▼
Embed Question → Vector
    │
    ▼
FAISS Similarity Search → Top 6 Chunks
    │
    ▼
Build Prompt (Context + Question)
    │
    ▼
Send to Ollama (tinyllama, local)
    │
    ▼
Return Answer to Frontend
```

---

## Project Structure

```
PDF_RAG_APP/
├── backend/
│   ├── main.py            ← FastAPI server, API endpoints
│   ├── rag_pipeline.py    ← PDF processing, embedding, FAISS, retrieval
│   └── requirements.txt   ← Python dependencies
└── frontend/
    └── index.html         ← Single-page UI served by FastAPI
```

---

## Code Explanation

### `backend/rag_pipeline.py`

This file handles all the heavy lifting of the RAG pipeline.

**Embedding model:**
```python
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
```
Loads a lightweight but powerful sentence embedding model locally. Converts text into 384-dimensional vectors that capture semantic meaning.

**In-memory vector store:**
```python
vector_store = {
    "index": None,
    "chunks": []
}
```
Holds the FAISS index and original chunk text in memory. Resets on server restart or new PDF upload.

**PDF text extraction:**
```python
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text
```
Uses PyMuPDF (`fitz`) to open the PDF from raw bytes and extract all text page by page into a single string.

**Chunking:**
```python
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
```
Splits the full text into overlapping chunks of 800 characters with 100-character overlap. Overlap ensures that sentences split across chunk boundaries don't lose context.

**Building the FAISS index:**
```python
def build_vector_store(chunks: list[str]):
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
```
Embeds all chunks into float32 vectors and adds them to a FAISS flat L2 index. `IndexFlatL2` does exact nearest-neighbour search using Euclidean distance — accurate and fast for small-to-medium documents.

**Retrieval:**
```python
def retrieve_relevant_chunks(query: str, top_k: int = 6) -> list[str]:
```
Embeds the user's question and searches the FAISS index for the 6 most similar chunks. These chunks form the context window passed to the LLM.

---

### `backend/main.py`

The FastAPI server with three endpoints.

**Serve frontend:**
```python
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
```
Reads and serves `frontend/index.html` directly from FastAPI. This means no separate frontend server is needed — everything runs on `http://127.0.0.1:8000`.

**PDF upload:**
```python
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
```
Accepts a multipart file upload, runs it through the full RAG pipeline (extract → chunk → embed → index), and returns a confirmation with the chunk count.

**Ask a question:**
```python
@app.post("/ask")
async def ask_question(payload: dict):
```
Accepts a JSON payload with a `question` field. Retrieves the top 6 relevant chunks, builds a prompt, and sends it to Ollama's local API at `http://localhost:11434/api/generate`. Returns both the answer and the source chunks used.

**Prompt design:**
```
You are a helpful assistant. Use the context below to answer the question.
Only use what's in the context. If the answer isn't there, say "I don't know".

Context:
{context}

Question: {question}
Answer:
```
Keeps the LLM grounded to the document. The "say I don't know" instruction reduces hallucination.

---

### `frontend/index.html`

A single-file UI with no framework dependencies.

**Key behaviours:**
- **Auto-upload** — PDF is uploaded immediately on file select, no extra button needed
- **Relative API calls** — `BASE_URL = ""` means all fetch calls go to the same origin as the page, avoiding CORS issues entirely
- **Thinking indicator** — animated dots appear while waiting for the LLM response
- **Source chunks sidebar** — after each answer, the left panel updates to show exactly which parts of the PDF the answer was based on
- **Enter to send** — Shift+Enter adds a new line, plain Enter submits the question

---

## Features

- **Fully local** — no OpenAI, no HuggingFace API, no data sent anywhere
- **Works on restricted networks** — ideal for office/work WiFi that blocks external AI APIs
- **PDF auto-indexing** — upload and it's ready instantly
- **Semantic search** — finds relevant chunks by meaning, not just keyword match
- **Source transparency** — see exactly which chunks from the PDF were used to answer
- **Clean dark UI** — minimal, distraction-free interface
- **Single server** — FastAPI serves both the API and the frontend

---

## Setup & Installation

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/download) installed and running
- `tinyllama` model pulled

### 1. Clone the repo

```bash
git clone https://github.com/aravindan112/PDF_RAG_APP.git
cd PDF_RAG_APP
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Install Ollama and pull tinyllama

Download Ollama from https://ollama.com/download, then:

```bash
ollama pull tinyllama
```

### 5. Run the app

```bash
cd backend
uvicorn main:app --reload
```

Open your browser at `http://127.0.0.1:8000`

---

## Usage

1. Open `http://127.0.0.1:8000` in your browser
2. Click the upload zone or drag and drop a PDF
3. Wait for the green "PDF processed" status
4. Type a question in the input bar and press Enter
5. The answer appears in the chat panel
6. The left sidebar shows which chunks from the PDF were used

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| PDF parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS (IndexFlatL2) |
| LLM | tinyllama via Ollama |
| HTTP client | httpx (async) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Why RAG?

LLMs have a fixed knowledge cutoff and can't read your files. RAG solves this by retrieving relevant text from your document at query time and injecting it into the prompt as context. The LLM never needs to "know" your document — it just reads the relevant parts on demand, every time you ask a question.

---

## Limitations

- Vector store is in-memory only — restarting the server requires re-uploading the PDF
- `tinyllama` is a small model — answers are reasonable but not as detailed as larger models
- One PDF at a time — uploading a new PDF replaces the previous index

---