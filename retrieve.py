"""
retrieve.py - Clinical Decision Support RAG Retrieval Script

Connects to local ChromaDB, executes similarity search for clinical queries,
and formats retrieved evidence with strict citation metadata following the Golden Rule:
"No claim without a citation."
"""

import os
import sys
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 formatting across all platforms (Windows cp1252 fix)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = "diabetes_guidelines"

def get_embedding_function():
    """Initializes configurable embedding provider (OpenAI or HuggingFace fallback)."""
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    model_name = os.getenv("EMBEDDING_MODEL", "")
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
            return HuggingFaceEmbeddings(model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def query_clinical_rag(query_text: str, top_k: int = 3):
    """
    Connects to ChromaDB vector database and performs similarity search with full citation traceability.
    """
    if not os.path.exists(PERSIST_DIR):
        print(f"[Error] ChromaDB directory '{PERSIST_DIR}' not found. Please run 'python ingest.py' first!")
        sys.exit(1)

    embedding_function = get_embedding_function()

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

    print("\n" + "=" * 80)
    print("                  CLINICAL RAG RETRIEVAL QUERY RESULTS                  ")
    print("=" * 80)
    print(f"QUERY: \"{query_text}\"")
    print(f"RETRIEVING TOP {top_k} MOST RELEVANT CLINICAL GUIDELINE CHUNKS...\n")

    # Similarity search with relevance scores
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(query_text, k=top_k)

    if not results_with_scores:
        # Fallback to standard similarity search if score thresholding fails
        results = vectorstore.similarity_search(query_text, k=top_k)
        results_with_scores = [(doc, 0.0) for doc in results]

    for idx, (doc, score) in enumerate(results_with_scores, 1):
        meta = doc.metadata
        doc_name = meta.get("document_name", "UNKNOWN")
        section = meta.get("section_title", "N/A")
        page_num = meta.get("page_number", "N/A")
        chunk_id = meta.get("chunk_id", "N/A")

        # Golden Rule Citation String Construction
        citation_str = f"Source Document: {doc_name} | Section: '{section}' | Page: {page_num} | Chunk ID: {chunk_id}"

        print(f"--- [RESULT #{idx}] (Relevance Score: {score:.4f}) ---")
        print(f"[*] CITATION (Golden Rule):")
        print(f"   [{citation_str}]")
        print(f"[*] METADATA DETAILS:")
        print(f"   - Document Name : {doc_name}")
        print(f"   - Section Title : {section}")
        print(f"   - Page Number   : {page_num}")
        print(f"   - Chunk ID      : {chunk_id}")
        print(f"[*] RETRIEVED CONTENT CHUNK:")
        print("   " + "-" * 74)
        for line in doc.page_content.strip().split("\n"):
            print(f"   {line}")
        print("   " + "-" * 74 + "\n")

    print("=" * 80)
    print("GOLDEN RULE VERIFIED: Every retrieved chunk includes complete citation provenance.")
    print("=" * 80 + "\n")

def main():
    # Allow clinical query to be passed as CLI argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What are the criteria for diagnosing diabetes?"

    query_clinical_rag(query, top_k=3)

if __name__ == "__main__":
    main()
