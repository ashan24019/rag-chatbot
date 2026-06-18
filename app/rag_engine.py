import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load a PDF or .txt file and return a list of Document objects.
def load_document(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf=8")
    else:
        raise ValueError("Unsupported file type. Only .pdf and .txt are supported.")
    
    return loader.load()



# Split documents into overlapping chunks.
def chunk_documents(documents, chunk_size: int = 500, chunk_overlap: int = 200):
    spillter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap = chunk_overlap,
        separators= ["\n\n", "\n", ". ", " ", ""]
    )

    return spillter.split_documents(documents)



# Convert each chunk into a vector and store in FAISS.
def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


# Save the FAISS index to disk.
def save_vector_store(vector_store, path: str = "vector_store"):
    vector_store.save_local(path)


#Load a previously saved FAISS index from disk.
def load_vector_store(path: str = "vector_store"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return FAISS.load_local(
        path, embeddings, allow_dangerous_deseriazation= True
    )



def ask(vector_store, question: str, k: int =3) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retriever = vector_store.as_retriever(search_kwargs = {"k": k})

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = retriever,
        return_source_documents = True
    )

    result = qa_chain.invoke({"query": question})

    return {
        "answer" : result["result"],
        "source" : [doc.page_content[: 200] for doc in result["source_documents"]]
    }