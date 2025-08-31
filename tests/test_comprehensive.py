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
from utils.document_ops import load_documents, load_excel_or_csv_as_documents

TEST_TEXT = "Hello World"

# -------------------------
# DocHandler Tests
# -------------------------
def test_doc_handler_save_and_read(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text(TEST_TEXT)

    dh = DocHandler(data_dir=str(tmp_path))

    class DummyUpload:
        name = str(file_path)
        def read(self): 
            return TEST_TEXT.encode()

    saved_path = dh.save_file(DummyUpload())
    assert Path(saved_path).exists()
    content = dh.read_file(saved_path)
    assert TEST_TEXT in content

def test_dochandler_unsupported_file(tmp_path):
    dh = DocHandler(data_dir=str(tmp_path))
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("data")
    with pytest.raises(Exception):
        dh.read_file(bad_file)

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def mock_faiss_manager():
    with patch("src.document_ingestion.data_ingestion.FaissManager") as fm:
        instance = fm.return_value
        instance.load_or_create.return_value = MagicMock(as_retriever=MagicMock())
        instance.add_documents.return_value = 1
        yield instance

@pytest.fixture
def chat_ingestor(tmp_path, mock_faiss_manager):
    ci = ChatIngestor(temp_base=str(tmp_path), faiss_base=str(tmp_path), use_session_dirs=False)
    return ci

# -------------------------
# Helper Functions
# -------------------------
def create_dummy_pdf(path: Path, num_pages=2, text="Hello PDF"):
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"{text} page {i+1}")
    doc.save(path)

# -------------------------
# ChatIngestor Tests
# -------------------------
# def test_chat_ingestor_build_retriever_csv(chat_ingestor, tmp_path):
#     csv_bytes = b"col1,col2\n1,a\n2,b"
#     upload_file = UploadFile(filename="dummy.csv", file=BytesIO(csv_bytes))
#     dummy_path = tmp_path / "dummy.csv"
#     dummy_path.write_bytes(csv_bytes)

#     with patch("utils.file_io.save_uploaded_files", return_value=[dummy_path]), \
#          patch("utils.document_ops.load_excel_or_csv_as_documents") as mock_loader:
#         mock_loader.return_value = [Document(page_content="row1", metadata={"row_id": 0})]
#         retriever = chat_ingestor.built_retriver([upload_file])

#     assert retriever is not None
#     assert mock_loader.called
def test_chat_ingestor_build_retriever_csv(chat_ingestor, tmp_path):
    # create a real CSV file
    csv_path = tmp_path / "dummy.csv"
    csv_path.write_text("col1,col2\n1,a\n2,b")

    # patch save_uploaded_files to return this path
    with patch("utils.file_io.save_uploaded_files", return_value=[csv_path]), \
         patch("utils.document_ops.load_excel_or_csv_as_documents") as mock_loader:
        
        mock_loader.return_value = [Document(page_content="row1", metadata={"row_id": 0})]
        retriever = chat_ingestor.built_retriver([MagicMock()])  # dummy UploadFile

    assert retriever is not None
    assert mock_loader.called

    assert retriever is not None
    assert mock_loader.called

def test_chat_ingestor_build_retriever_xlsx(chat_ingestor, tmp_path):
    df = pd.DataFrame({"A":[1,2], "B":["x","y"]})
    xlsx_bytes = BytesIO()
    df.to_excel(xlsx_bytes, index=False)
    xlsx_bytes.seek(0)
    upload_file = UploadFile(filename="dummy.xlsx", file=xlsx_bytes)
    dummy_path = tmp_path / "dummy.xlsx"
    with open(dummy_path, "wb") as f:
        f.write(xlsx_bytes.getvalue())

    with patch("utils.file_io.save_uploaded_files", return_value=[dummy_path]), \
         patch("utils.document_ops.load_excel_or_csv_as_documents") as mock_loader:
        mock_loader.return_value = [Document(page_content="row1", metadata={"row_id": 0})]
        retriever = chat_ingestor.built_retriver([upload_file])

    assert retriever is not None
    assert mock_loader.called

def test_chat_ingestor_build_retriever_txt(chat_ingestor, tmp_path):
    txt_bytes = b"Hello world"
    upload_file = UploadFile(filename="dummy.txt", file=BytesIO(txt_bytes))
    dummy_path = tmp_path / "dummy.txt"
    dummy_path.write_bytes(txt_bytes)

    with patch("utils.file_io.save_uploaded_files", return_value=[dummy_path]), \
         patch("utils.document_ops.load_documents") as mock_loader:
        mock_loader.return_value = [Document(page_content="row1", metadata={"row_id": 0})]
        retriever = chat_ingestor.built_retriver([upload_file])

    assert retriever is not None
    assert mock_loader.called

def test_chat_ingestor_build_retriever_pdf(chat_ingestor, tmp_path):
    pdf_path = tmp_path / "dummy.pdf"
    create_dummy_pdf(pdf_path)

    upload_file = UploadFile(filename="dummy.pdf", file=BytesIO(pdf_path.read_bytes()))

    with patch("utils.file_io.save_uploaded_files", return_value=[pdf_path]), \
         patch("utils.document_ops.load_documents") as mock_loader:
        mock_loader.return_value = [Document(page_content="PDF content", metadata={"source": str(pdf_path), "row_id": 0})]
        retriever = chat_ingestor.built_retriver([upload_file])

    assert retriever is not None
    assert mock_loader.called

def test_chat_ingestor_build_retriever_multi_page_pdf(chat_ingestor, tmp_path):
    pdf_path = tmp_path / "multi_page.pdf"
    create_dummy_pdf(pdf_path, num_pages=3)

    upload_file = UploadFile(filename="multi_page.pdf", file=BytesIO(pdf_path.read_bytes()))

    with patch("utils.file_io.save_uploaded_files", return_value=[pdf_path]), \
         patch("utils.document_ops.load_documents") as mock_loader:
        mock_loader.return_value = [
            Document(page_content=f"PDF content page {i+1}", metadata={"source": str(pdf_path), "row_id": i})
            for i in range(3)
        ]
        retriever = chat_ingestor.built_retriver([upload_file])

    assert retriever is not None
    assert mock_loader.called
