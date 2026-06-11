from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import os
from rag_pipeline import extract_text_from_pdf, chunk_text, build_vector_store, retrieve_relevant_chunks

app = FastAPI()

# Serves the frontend HTML file at http://127.0.0.1:8000
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    with open(frontend_path, "r", encoding="utf-8") as f:
        return f.read()

# Accepts PDF upload, extracts text, chunks it, builds FAISS vector store
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    text = extract_text_from_pdf(pdf_bytes)
    chunks = chunk_text(text)
    build_vector_store(chunks)
    return JSONResponse({"message": f"PDF processed. {len(chunks)} chunks indexed."})

# Accepts a question, retrieves relevant chunks, sends to Ollama, returns answer + chunks
@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question", "")
    chunks = retrieve_relevant_chunks(question)
    context = "\n\n".join(chunks)

    prompt = f"""You are a helpful assistant. Use the context below to answer the question.
Only use what's in the context. If the answer isn't there, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

    # Calls Ollama running locally on default port 11434
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": prompt,
                "stream": False
            }
        )

    result = response.json()
    answer = result.get("response", "No response from model.")

    # Return both the answer and the chunks used to generate it
    return JSONResponse({"answer": answer, "context": chunks})