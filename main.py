from dotenv import load_dotenv

from app.rag_engine import (
    ask,
    build_vector_store,
    chunk_documents,
    load_document,
    save_vector_store,
)

load_dotenv()

def main():
    print("="*50)
    print("  RAG Chatbot")
    print("="*50)

    file_path = input("\nPath to your PDF or TXT file: ").strip()

    print("\n[1/3] Loading documents...")
    docs = load_document(file_path)
    print(f"      {len(docs)} page(s) loaded")

    print("[2/3] Chunking...")
    chunks = chunk_documents(docs)
    print(f"      {len(chunks)} chunks created")

    print("[3/3] Embedding and indexing...")
    print("      (This calls the OpenAI API — takes 10-30 seconds)")
    vs = build_vector_store(chunks)

    save_vector_store(vs)
    print("      Done. Index saved to ./vector_store/")

    print("\nReady. Ask questions about your document.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if not question:
            continue

        result = ask(vs, question)

        print(f"\nBot: {result['answer']}")
        print("\nRetrieved from:")
        for i, src in enumerate(result["source"], 1):
            print(f"  [{i}] {src[:150]}...")
        print()

if __name__ == "__main__":
    main()
