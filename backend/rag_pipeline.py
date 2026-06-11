import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Same embedding model — this part works fine, no reason to change it

vector_store = {
    "index": None,
    "chunks": []
}


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text
# Opens PDF from raw bytes, loops every page, pulls all text into one string


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
# Increased chunk_size to 800 and overlap to 100
# Bigger chunks = more context per retrieval = better answers from the LLM


def build_vector_store(chunks: list[str]):
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    vector_store["index"] = index
    vector_store["chunks"] = chunks
# Embeds all chunks into vectors and stores in FAISS — unchanged


def retrieve_relevant_chunks(query: str, top_k: int = 6) -> list[str]:
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    _, indices = vector_store["index"].search(query_embedding, top_k)
    results = [vector_store["chunks"][i] for i in indices[0]]
    return results
# Increased top_k to 6 — gives the LLM more context to work with