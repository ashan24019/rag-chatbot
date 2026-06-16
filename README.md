# RAG Chatbot — Project 1

Ask questions about any PDF or text document. Built with LangChain, OpenAI, and FAISS.

## Stack
Python 3.11 · LangChain · OpenAI API · FAISS · pytest · Docker · GitHub Actions

## Local setup

\```bash
# Clone
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

# Create and activate venv
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# Run
python main.py
\```

## Run tests
\```bash
pytest tests/ -v --cov=app
\```

## Project structure
\```
rag-chatbot/
├── app/
│   ├── __init__.py
│   └── rag_engine.py      # all AI logic — loading, chunking, embedding, QA
├── tests/
│   └── test_rag_engine.py
├── .github/workflows/
│   └── ci.yml
├── main.py                # CLI entrypoint only
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
\```