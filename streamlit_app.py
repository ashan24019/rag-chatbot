import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from app.rag_engine import (
    ask,
    build_vector_store,
    chunk_documents,
    load_document,
)

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 RAG Chatbot")
st.caption("Upload a PDF or TXT file and ask questions about it.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None      # FAISS index for current doc

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []        # list of {role, content, source}

if "current_file" not in st.session_state:
    st.session_state.current_file = None      # name of the uploaded file


uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt"],
    help="PDF or plain text files only",
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.current_file:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                docs = load_document(tmp_path)
                chunks = chunk_documents(docs)
                st.session_state.vector_store = build_vector_store(chunks)
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []
                st.success(
                    f"✅ Indexed {uploaded_file.name} — "
                    f"{len(docs)} pages, {len(chunks)} chunks"
                )
            except Exception as e:
                st.error(f"Failed to process file: {e}")
            finally:
                os.unlink(tmp_path)

st.divider()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("source"):
            with st.expander("📄 View source"):
                for i, src in enumerate(message["source"], 1):
                    st.caption(f"[{i}] {src}...")

question = st.chat_input(
    "Ask a question about your document...",
    disabled=st.session_state.vector_store is None,
)

if question:
    with st.chat_message("user"):
        st.write(question)

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "source": [],
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(st.session_state.vector_store, question)
                st.write(result["answer"])

                if result["source"]:
                    with st.expander("📄 View source"):
                        for i, src in enumerate(result["source"], 1):
                            st.caption(f"[{i}] {src}...")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "source": result["source"],
                })

            except Exception as e:
                st.error(f"Error: {e}")

if st.session_state.vector_store is None:
    st.info("👆 Upload a document above to start chatting.")
