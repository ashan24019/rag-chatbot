import os
import tempfile
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.rag_engine import ask, build_vector_store, chunk_documents, load_document
from app.schemas import AskRequest, AskResponse, UploadResponse
from app.session_store import create_session, get_session

load_dotenv()

app = FastAPI(
    title="RAG Chatbot API",
    description="Upload a document, then ask questions about it.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to specific domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only .pdf and .txt are supported.",
        )
    
    suffix = ".pdf" if file.filename.endswith(".pdf") else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = load_document(tmp_path)
        chunks = chunk_documents(docs)
        vector_store = build_vector_store(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")
    finally:
        # Always clean up the temp file — this answers your Sprint 2 Q3 directly:
        # skipping this on a server handling 1000 uploads/day fills the disk
        # with orphaned temp files until the server runs out of storage and crashes.
        os.unlink(tmp_path)

    session_id = str(uuid.uuid4())
    create_session(session_id, vector_store)

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        pages=len(docs),
        chunks=len(chunks),
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Answer a question using the vector store tied to the given session_id.
    """
    vector_store = get_session(request.session_id)

    if vector_store is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Upload a document first via /upload.",
        )

    try:
        result = ask(vector_store, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}")

    return AskResponse(answer=result["answer"], sources=result["sources"])