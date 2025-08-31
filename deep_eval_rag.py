import sys
import os
from pathlib import Path
from src.document_ingestion.data_ingestion import ChatIngestor
from src.document_chat.retrieval import ConversationalRAG
from utils.config_loader import load_config
from utils.model_loader import ApiKeyManager,EvalModelLoader
import pytest
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import json
from dotenv import load_dotenv
load_dotenv()

api_keys_str = os.getenv("API_KEYS")
if not api_keys_str:
    raise ValueError("API_KEYS not found in environment")

API_KEYS = json.loads(api_keys_str)

FAISS_INDEX_PATH = Path("faiss_index")

def test_conversational_rag_on_pdf(pdf_path: str, question: str):
    try:
        session_id = "test_conversational_rag"
        session_index_path = FAISS_INDEX_PATH / session_id

        # Initialize ingestor
        ingestor = ChatIngestor(session_id=session_id)

        # Step 1: Check if FAISS index exists
        if session_index_path.exists() and any(session_index_path.iterdir()):
            print("Loading existing FAISS index...")
        else:
            print("FAISS index not found. Ingesting PDF and creating index...")
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file does not exist at: {pdf_path}")

            # Ingest PDF and build FAISS index
            with open(pdf_path, "rb") as f:
                uploaded_files = [f]
                ingestor.built_retriver(uploaded_files)

        # Step 2: Initialize Conversational RAG and load retriever + chain
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_path=str(session_index_path), k=2)

        # Step 3: Ask the question
        response = rag.invoke(question, chat_history=[])
        print(f"\nQuestion: {question}\nAnswer: {response}")

        retrieved_docs = rag.retriever.get_relevant_documents(question)
        context_texts = [d.page_content for d in retrieved_docs]
        print ("Context Text is",context_texts)


        test_case = LLMTestCase(
            input=question,
            actual_output=response,
            #expected_output=None,  
            retrieval_context=context_texts
        )
        
        # correctness_metric = GEval(
        #     name="Correctness",
        #     criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        #     threshold=0.5
        # )
        # print (correctness_metric)
        

        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
        
        #assert_test(test_case, metrics)
        evaluate([test_case], [answer_relevancy_metric])

    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    pdf_path = "data/NIPS-2017-attention-is-all-you-need-Paper.pdf"
    question = "What is the significance of the attention mechanism?"

    test_conversational_rag_on_pdf(pdf_path, question)
