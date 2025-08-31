from __future__ import annotations
from pathlib import Path
from typing import Iterable, List,Dict,Any
from fastapi import UploadFile
from langchain.schema import Document
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader,CSVLoader,UnstructuredExcelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".pptx", ".csv", ".xlsx", ".xls", ".md"
}


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load docs using appropriate loader based on extension."""
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext in {".txt", ".md"}:
                loader = TextLoader(str(p), encoding="utf-8")
            elif ext == ".pptx":
                loader = UnstructuredPowerPointLoader(str(p))
            elif ext == ".csv":
                loader = CSVLoader(str(p), encoding="utf-8")  
            elif ext in {".xlsx", ".xls"}:
                loader = UnstructuredExcelLoader(str(p)) 
            else:
                log.warning("Unsupported extension skipped", path=str(p))
                continue
            docs.extend(loader.load())
        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e

def concat_for_analysis(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        parts.append(f"\n--- SOURCE: {src} ---\n{d.page_content}")
    return "\n".join(parts)

def concat_for_comparison(ref_docs: List[Document], act_docs: List[Document]) -> str:
    left = concat_for_analysis(ref_docs)
    right = concat_for_analysis(act_docs)
    return f"<<REFERENCE_DOCUMENTS>>\n{left}\n\n<<ACTUAL_DOCUMENTS>>\n{right}"

# ---------- Helpers ----------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile -> .name + .getbuffer() API"""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename
    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()

def read_file_via_handler(handler, path: str) -> str:
    if hasattr(handler, "read_file"):
        return handler.read_file(path)  # type: ignore
    if hasattr(handler, "read_"):
        return handler.read_(path)  # type: ignore
    raise RuntimeError("DocHandler has neither read_pdf nor read_ method.")




def load_documents_from_sql(
    connection_string: str,
    query: str,
    metadata_fields: Dict[str, Any] = None
) -> List[Document]:
    """
    Load rows from any SQL database and convert them to langchain Documents.
    
    Args:
        connection_string: SQLAlchemy compatible DB URL (e.g., 'postgresql://user:pass@host/db')
        query: SQL query to fetch data
        metadata_fields: Optional dict of additional metadata to attach to each Document
    
    Returns:
        List[Document]
    """
    df = pd.read_sql(query, create_engine(connection_string))
    docs = []
    for idx, row in df.iterrows():
        text = " | ".join([str(v) for v in row.values])  
        metadata = {"row_id": idx}
        if metadata_fields:
            metadata.update(metadata_fields)
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def load_excel_or_csv_as_documents(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".xlsx":
            df = pd.read_excel(path, engine="openpyxl")
        # elif ext == ".xls":
        #     try:
        #         import xlrd  
        #     except ImportError:
        #         raise RuntimeError(
        #             "Install xlrd==1.2.0 to support .xls files, or convert to .xlsx"
        #         )
        #     df = pd.read_excel(path, engine="xlrd")
        else:
            return []

        docs = []
        for idx, row in df.iterrows():
            text = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
            metadata = {"source": str(path), "row_id": idx}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    except Exception as e:
        log.error("Failed to load Excel/CSV as documents", error=str(e), file=str(path))
        return []