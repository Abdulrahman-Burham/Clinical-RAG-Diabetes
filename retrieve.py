"""
retrieve.py - Multi-Document Clinical Guidelines Vector Retrieval & Cohere Reranking Pipeline

Queries local ChromaDB vector store containing ingested clinical diabetes guidelines,
applies optional Cohere Reranking (rerank-v3.5) for state-of-the-art precision,
enforces strict Golden Rule traceability, displays complete chunk metadata,
supports document-level metadata filtering and query normalization/expansion for Arabic typos.
"""

import os
import sys
import re
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

def normalize_arabic_query(query_text: str) -> str:
    """Normalizes common Arabic spelling typos and expands broad terms."""
    q = query_text.strip()
    
    # Correct common typos
    q = re.sub(r'\bالسسكر\b', 'السكري', q)
    q = re.sub(r'\bسسكر\b', 'سكر', q)
    q = re.sub(r'\bسس\b', 'س', q)
    q = re.sub(r'[أإآ]', 'ا', q)
    q = re.sub(r'ى\b', 'ي', q)
    
    return q

def get_embedding_function():
    """Initializes configurable embedding provider (Multilingual HuggingFace or OpenAI)."""
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    openai_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    if provider == "openai" and openai_key and openai_key != "your_openai_api_key_here":
        print(f"Using OpenAI Embeddings ({model_name or 'text-embedding-3-small'})...")
        from langchain_openai import OpenAIEmbeddings
        base_url = "https://api.openai.com/v1" if "openrouter" in api_base else api_base
        return OpenAIEmbeddings(
            model=model_name or "text-embedding-3-small",
            openai_api_key=openai_key,
            openai_api_base=base_url
        )
    else:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)

def load_index(persist_dir: str = None):
    """Loads existing ChromaDB vectorstore."""
    if persist_dir is None:
        persist_dir = PERSIST_DIR
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    embedding_function = get_embedding_function()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

def retrieve(vectordb, query_text: str, k: int = 3, doc_filter: str = None):
    """
    Retrieves top k documents with optional Cohere Reranking for high precision.
    """
    filter_dict = {}
    if doc_filter and doc_filter != "ALL" and not doc_filter.startswith("All") and not doc_filter.startswith("جميع"):
        filter_dict = {"document_name": doc_filter}
    
    search_query = normalize_arabic_query(query_text)
    
    # Retrieve top candidates for reranking
    candidate_k = max(k * 3, 10)
    try:
        if filter_dict:
            initial_results = vectordb.similarity_search_with_score(search_query, k=candidate_k, filter=filter_dict)
        else:
            initial_results = vectordb.similarity_search_with_score(search_query, k=candidate_k)
    except Exception:
        return []

    if not initial_results:
        return []

    # Check for Cohere API key for Reranking
    cohere_key = os.getenv("COHERE_API_KEY", "")
    if cohere_key:
        try:
            import cohere
            co = cohere.ClientV2(cohere_key)
            docs_text = [doc.page_content for doc, _ in initial_results]
            
            res = co.rerank(
                model="rerank-v3.5",
                query=query_text,
                documents=docs_text,
                top_n=k
            )
            
            reranked = []
            for r in res.results:
                orig_doc, _ = initial_results[r.index]
                score = float(r.relevance_score)
                reranked.append((orig_doc, score))
            return reranked
        except Exception as e:
            pass

    # Standard normalized similarity scores fallback
    normalized_results = []
    for doc, dist in initial_results[:k]:
        score = 1.0 / (1.0 + max(0.0, float(dist)))
        normalized_results.append((doc, score))
        
    return normalized_results

def query_clinical_rag(query_text: str, top_k: int = 3, doc_filter: str = None):
    """
    Queries local ChromaDB collection and returns top-K evidence chunks with distance/rerank scores.
    """
    if not os.path.exists(PERSIST_DIR):
        print(f"[Error] ChromaDB directory '{PERSIST_DIR}' does not exist. Please run 'python ingest.py' first.")
        return []

    vectordb = load_index(PERSIST_DIR)

    search_query = normalize_arabic_query(query_text)

    print("\n" + "=" * 80)
    print("      MULTI-DOCUMENT CLINICAL RAG RETRIEVAL & COHERE RERANK RESULTS       ")
    print("=" * 80)
    print(f"RAW QUERY: \"{query_text}\" | SEARCH QUERY: \"{search_query}\"")
    if doc_filter:
        print(f"FILTER: Document Name == '{doc_filter}'")
    print(f"RETRIEVING & RERANKING TOP {top_k} MOST RELEVANT CLINICAL EVIDENCE CHUNKS...\n")

    results_with_scores = retrieve(vectordb, query_text, k=top_k, doc_filter=doc_filter)

    for idx, (doc, score) in enumerate(results_with_scores, 1):
        meta = doc.metadata
        print(f"--- [RESULT {idx}/{len(results_with_scores)}] Relevance / Cohere Rerank Score: {score:.4f} ---")
        print(f"  📌 Document:  {meta.get('document_name', 'N/A')}")
        print(f"  📖 Section:   {meta.get('section_title', 'N/A')}")
        print(f"  📄 Page Num:  {meta.get('page_number', 'N/A')}")
        print(f"  🆔 Chunk ID:  {meta.get('chunk_id', 'N/A')}")
        print(f"  📝 Content Snippet:")
        content_preview = doc.page_content.strip()
        if len(content_preview) > 300:
            content_preview = content_preview[:300] + "..."
        print(f"     \"{content_preview}\"\n")

    return results_with_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Clinical Diabetes Guidelines Vector Database")
    parser.add_argument("query", type=str, help="Clinical question or query text")
    parser.add_argument("--k", type=int, default=3, help="Number of top evidence chunks to retrieve (default: 3)")
    parser.add_argument("--doc", type=str, default=None, help="Filter search to specific document name")

    args = parser.parse_args()
    query_clinical_rag(args.query, top_k=args.k, doc_filter=args.doc)
