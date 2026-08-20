from fastapi import FastAPI, UploadFile, File
from PyPDF2 import PdfReader

from chunking import chunk_text
from embedding_service import generate_embedding
from vector_store import store_embeddings, search_embeddings
from rag_service import generate_answer

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "RAG IT API is running"
    }

@app.post("/upload")
def upload_resume(file: UploadFile = File(...)):
    reader = PdfReader(file.file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Step 1: Split text into chunks
    chunks = chunk_text(text)

    # Step 2: Generate embeddings
    embeddings = []

    for chunk in chunks:
        vector = generate_embedding(chunk)
        embeddings.append(vector)

    # Step 3: Store chunks + embeddings in ChromaDB
    store_embeddings(chunks, embeddings)

    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "message": "Resume processed and stored successfully"
    }

@app.post("/search")
def search_resume(question: str):
    query_embedding = generate_embedding(question)

    results = search_embeddings(query_embedding)

    relevant_chunks = results["documents"][0]

    context = "\n\n".join(relevant_chunks)

    answer = generate_answer(
        question,
        context
    )

    return {
        "question": question,
        "answer": answer,
        "sources": relevant_chunks
    }