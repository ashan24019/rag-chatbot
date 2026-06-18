import pytest
from langchain.schema import Document

from app.rag_engine import chunk_documents, load_document


def make_doc(text: str) -> Document:
    """Helper — wraps raw text in a LangChain Document."""
    return Document(page_content=text, metadata={"source": "test"})


class TestChucking:
    def test_long_text_produces_multiple_chunks(self):
        """A document longer than chunk_size must be split."""

        docs = [make_doc("This is a sentence. " * 100)]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1

    def test_short_text_stays_as_one_chunk(self):
        """A short document should not be split."""
        docs = [make_doc("Short text.")]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1

    def test_chunk_size_respected(self):
        """No chunk should be significantly larger than chunk_size."""
        docs = [make_doc("word "*500)]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
        for chunk in chunks:
            assert len(chunk.page_content) <= 300  # tolerance for separator logic


    def test_overlap_shares_content_between_chunks(self):
        """Consecutive chunks should share some content due to overlap."""

        docs = [make_doc("Alpha beta gamma delta. " * 80)]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=100)

        if len(chunks) > 1:
            tail_of_first = chunks[0].page_content[-60:]
            head_of_second = chunks[1].page_content[:120]
            shared = any(w in head_of_second for w in tail_of_first.split()[:4])
            assert shared


class TestLoading:
    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_document("does_not_exist.txt")

    def test_unsupported_extension_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document("notes.docx")
