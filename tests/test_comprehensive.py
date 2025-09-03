import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*TemplateResponse.*name is not the first parameter.*"
)

import pytest
import fitz
from pathlib import Path
from io import BytesIO
from unittest.mock import MagicMock, patch
import pandas as pd
from fastapi import UploadFile
from langchain.schema import Document
from langchain_core.outputs import Generation

from src.document_ingestion.data_ingestion import DocHandler, ChatIngestor, FaissManager
from src.document_chat.retrieval import ConversationalRAG
from exception.custom_exception import DocumentPortalException
from utils.document_ops import load_documents, load_excel_or_csv_as_documents
from src.document_analyzer.data_analysis import DocumentAnalyzer

# ---------- Initialization ----------
def test_init_success():
    rag = ConversationalRAG(session_id="test")
    assert rag.llm is not None
    assert rag.qa_prompt is not None
    assert rag.contextualize_prompt is not None

#The test ensures that initialization errors in LLM loading are handled properly
@patch("src.document_chat.retrieval.ConversationalRAG._load_llm", side_effect=Exception("LLM error"))
def test_init_failure(mock_load_llm):
    with pytest.raises(DocumentPortalException):
        ConversationalRAG(session_id="test")

# Test for retriever
@patch("os.path.isdir", return_value=True)
@patch("src.document_chat.retrieval.FAISS.load_local")
@patch("utils.model_loader.ModelLoader.load_embeddings")
def test_load_retriever_success(mock_load_local, mock_load_embeddings, mock_isdir):
    mock_load_embeddings.return_value = MagicMock()
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()
    mock_load_local.return_value = mock_vectorstore

    rag = ConversationalRAG(session_id="test")
    retriever = rag.load_retriever_from_faiss("dummy_path")

    assert retriever is not None
    assert rag.retriever is not None
    assert rag.chain is not None

# If FAISS index path is unavailable then the method should fail
def test_load_retriever_failure():
    rag = ConversationalRAG("test")
    with pytest.raises(DocumentPortalException):
        rag.load_retriever_from_faiss("non_existent_path")


# ---------- Chain ----------
#Checking how the class behaves if I try to build the LCEL chain without setting up a retriever first
def test_build_chain_without_retriever():
    rag = ConversationalRAG("test")
    with pytest.raises(DocumentPortalException):
        rag._build_lcel_chain()


# ---------- Invoke ----------

def test_invoke_success():
    # Mock the LCEL chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "mocked answer"

    # Mock the memory to return empty chat history
    mock_memory = MagicMock()
    mock_memory.load_memory_variables.return_value = {"chat_history": []}

    # Initialize the ConversationalRAG instance
    rag = ConversationalRAG("test_session")
    rag.chain = mock_chain
    rag.memory = mock_memory

    # Call invoke
    answer = rag.invoke("What is RAG?")

    # Assertions
    assert answer == "mocked answer"
    mock_chain.invoke.assert_called_once_with({
        "input": "What is RAG?",
        "chat_history": []
    })
    mock_memory.load_memory_variables.assert_called_once_with({})


def test_invoke_without_chain():
    rag = ConversationalRAG("test")
    with pytest.raises(DocumentPortalException):
        rag.invoke("Hello")


# ---------- Format Docs ----------
def test_format_docs_with_page_content():
    class Doc:
        def __init__(self, content): self.page_content = content

    rag = ConversationalRAG("test")
    docs = [Doc("doc1"), Doc("doc2")]
    result = rag._format_docs(docs)
    assert "doc1" in result and "doc2" in result

# Document Analyzer
    
def test_document_analyzer_init_failure():
# Simulate ModelLoader raising exception
    with patch("src.document_analyzer.data_analysis.ModelLoader") as MockLoader:
        MockLoader.side_effect = Exception("Load failed")
        with pytest.raises(DocumentPortalException) as exc_info:
            DocumentAnalyzer()
        assert "Error in DocumentAnalyzer initialization" in str(exc_info.value)
def test_document_analyzer_init_success():
    from src.document_analyzer.data_analysis import DocumentAnalyzer
    from unittest.mock import patch, MagicMock

    with patch("src.document_analyzer.data_analysis.ModelLoader") as MockLoader, \
         patch("src.document_analyzer.data_analysis.JsonOutputParser") as MockParser, \
         patch("src.document_analyzer.data_analysis.OutputFixingParser") as MockFixingParser, \
         patch("src.document_analyzer.data_analysis.PROMPT_REGISTRY", {"document_analysis": MagicMock()}):

        # Mock LLM and parsers
        MockLoader.return_value.load_llm.return_value = MagicMock()
        MockParser.return_value = MagicMock()
        MockFixingParser.from_llm.return_value = MagicMock()

        analyzer = DocumentAnalyzer()
        assert analyzer.loader is not None
        assert analyzer.llm is not None
        assert analyzer.parser is not None
        assert analyzer.fixing_parser is not None
        assert analyzer.prompt is not None



