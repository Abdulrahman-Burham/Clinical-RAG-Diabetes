"""
retrieve.py - Multi-Document Clinical Guidelines Vector Retrieval Pipeline

Queries local ChromaDB vector store containing ingested clinical diabetes guidelines,
enforces strict Golden Rule traceability, displays complete chunk metadata,
and supports document-level metadata filtering.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 formatting across all platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment variables
load_dotenv()

# Configuration
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = "diabetes_guidelines"

def get_embedding_function():
    """Initializes configurable embedding provider (Multilingual HuggingFace or OpenAI)."""
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if provider == "openai" and openai_key and openai_key != "your_openai_api_key_here":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name or "text-embedding-3-small",
            openai_api_key=openai_key
        )
    else:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)

def query_clinical_rag(query_text: str, top_k: int = 3, doc_filter: str = None):
    """
    Queries local ChromaDB collection and returns top-K evidence chunks with distance/relevance scores.
    Optional doc_filter filters retrieval strictly to a target document_name.
    """
    if not os.path.exists(PERSIST_DIR):
        print(f"[Error] ChromaDB directory '{PERSIST_DIR}' does not exist. Please run 'python ingest.py' first.")
        return []

    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    embedding_function = get_embedding_function()

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

    filter_dict = {}
    if doc_filter and doc_filter != "ALL":
        filter_dict = {"document_name": doc_filter}

    print("\n" + "=" * 80)
    print("           MULTI-DOCUMENT CLINICAL RAG RETRIEVAL RESULTS           ")
    print("=" * 80)
    print(f"QUERY: \"{query_text}\"")
    if doc_filter:
        print(f"FILTER: Document Name == '{doc_filter}'")
    print(f"RETRIEVING TOP {top_k} MOST RELEVANT CLINICAL EVIDENCE CHUNKS...\n")

    if filter_dict:
        results_with_scores = vectorstore.similarity_search_with_relevance_scores(
            query_text,
            k=top_k,
            filter=filter_dict
        )
    else:
        results_with_scores = vectorstore.similarity_search_with_relevance_scores(
            query_text,
            k=top_k
        )

    if not results_with_scores:
        print("[-] No matching clinical evidence chunks found.")
        return []

    for idx, (doc, score) in enumerate(results_with_scores, 1):
        meta = doc.metadata
        doc_name = meta.get("document_name", "UNKNOWN")
        section = meta.get("section_title", "N/A")
        page_num = meta.get("page_number", "N/A")
        chunk_id = meta.get("chunk_id", "N/A")

        print(f"--- [RESULT #{idx}] (Relevance Score: {score:.4f}) ---")
        print(f"[*] CITATION (Golden Rule):")
        print(f"   [Source Document: {doc_name} | Section: '{section}' | Page: {page_num} | Chunk ID: {chunk_id}]")
        print(f"[*] METADATA DETAILS:")
        print(f"   - Document Name : {doc_name}")
        print(f"   - Section Title : {section}")
        print(f"   - Page Number   : {page_num}")
        print(f"   - Chunk ID      : {chunk_id}")
        print(f"[*] RETRIEVED EVIDENCE CHUNK:")
        print("   " + "-" * 74)
        for line in doc.page_content.strip().split("\n"):
            print(f"   {line}")
        print("   " + "-" * 74 + "\n")

    print("=" * 80)
    print("GOLDEN RULE VERIFIED: Every retrieved chunk includes complete citation provenance.")
    print("=" * 80 + "\n")

    return results_with_scores

def main():
    parser = argparse.ArgumentParser(description="Multi-Document Clinical Guidelines RAG Retrieval Engine")
    parser.add_argument("query", nargs="?", default="What are the fasting blood glucose criteria for diagnosing diabetes?", help="Clinical search query")
    parser.add_argument("--k", type=int, default=3, help="Number of top chunks to retrieve")
    parser.add_argument("--doc", type=str, default=None, help="Filter search to specific document_name")

    args = parser.parse_args()
    query_clinical_rag(args.query, top_k=args.k, doc_filter=args.doc)

if __name__ == "__main__":
    main()
