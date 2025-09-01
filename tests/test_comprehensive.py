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

from src.document_ingestion.data_ingestion import DocHandler, ChatIngestor, FaissManager
from src.document_chat.retrieval import ConversationalRAG
from exception.custom_exception import DocumentPortalException
from utils.document_ops import load_documents, load_excel_or_csv_as_documents

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

