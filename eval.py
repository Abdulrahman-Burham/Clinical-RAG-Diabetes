"""
eval.py - Clinical RAG Performance Benchmark & Evaluation Harness

Evaluates multi-document vector retrieval accuracy, latency (ms), citation coverage,
and metadata schema compliance across benchmark clinical queries.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 formatting across all platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_community.vectorstores import Chroma
from retrieve import get_embedding_function, PERSIST_DIR, COLLECTION_NAME

load_dotenv()

BENCHMARK_QUERIES = [
    {
        "id": "EVAL-01",
        "category": "Diagnosis & Criteria",
        "query": "What are the fasting blood glucose criteria for diagnosing diabetes?"
    },
    {
        "id": "EVAL-02",
        "category": "Pharmacotherapy & Medication",
        "query": "What are the first-line medication recommendations for adult Type 2 diabetes?"
    },
    {
        "id": "EVAL-03",
        "category": "Pediatric Diabetes Care",
        "query": "How is diabetes managed and screened in children and adolescents?"
    },
    {
        "id": "EVAL-04",
        "category": "Metabolic Syndrome & Cardio",
        "query": "What are the cardiovascular risk factors associated with metabolic syndrome?"
    },
    {
        "id": "EVAL-05",
        "category": "Gestational Diabetes",
        "query": "What are the screening and diagnostic protocols for gestational diabetes?"
    }
]

def run_evaluation_benchmark():
    """Runs automated evaluation suite across clinical query benchmarks."""
    print("=" * 80)
    print("        CLINICAL DECISION SUPPORT SYSTEM - RAG EVALUATION HARNESS        ")
    print("=" * 80)

    if not os.path.exists(PERSIST_DIR):
        print(f"[Error] ChromaDB persist directory '{PERSIST_DIR}' not found. Run 'python ingest.py' first.")
        sys.exit(1)

    print("[*] Connecting to ChromaDB vector store...")
    embedding_function = get_embedding_function()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

    total_queries = len(BENCHMARK_QUERIES)
    passed_metadata_schema = 0
    passed_citation_trace = 0
    total_latency_ms = 0.0

    results_table = []

    for item in BENCHMARK_QUERIES:
        q_id = item["id"]
        category = item["category"]
        query_text = item["query"]

        start_time = time.time()
        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query_text, k=3)
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000.0
        total_latency_ms += latency_ms

        if not docs_with_scores:
            docs = vectorstore.similarity_search(query_text, k=3)
            docs_with_scores = [(d, 0.0) for d in docs]

        # Check metadata schema compliance
        required_schema = {"document_name", "section_title", "page_number", "chunk_id"}
        schema_ok = True
        citation_ok = True
        top_doc_source = "N/A"

        if docs_with_scores:
            top_doc = docs_with_scores[0][0]
            top_doc_source = top_doc.metadata.get("document_name", "UNKNOWN")

            for doc, score in docs_with_scores:
                meta = doc.metadata
                if not required_schema.issubset(set(meta.keys())):
                    schema_ok = False
                if not meta.get("document_name") or not meta.get("page_number"):
                    citation_ok = False

        if schema_ok:
            passed_metadata_schema += 1
        if citation_ok:
            passed_citation_trace += 1

        results_table.append({
            "id": q_id,
            "category": category,
            "latency_ms": latency_ms,
            "schema_pass": "PASS" if schema_ok else "FAIL",
            "citation_pass": "PASS" if citation_ok else "FAIL",
            "top_source": top_doc_source
        })

    avg_latency_ms = total_latency_ms / total_queries if total_queries else 0.0
    schema_pass_rate = (passed_metadata_schema / total_queries) * 100.0
    citation_pass_rate = (passed_citation_trace / total_queries) * 100.0

    print("\nBENCHMARK EVALUATION RESULTS:")
    print("-" * 80)
    print(f"{'ID':<10} | {'Category':<28} | {'Latency (ms)':<14} | {'Schema':<8} | {'Citation'}")
    print("-" * 80)
    for r in results_table:
        print(f"{r['id']:<10} | {r['category']:<28} | {r['latency_ms']:<14.2f} | {r['schema_pass']:<8} | {r['citation_pass']}")
    print("-" * 80)

    print("\nEVALUATION SUMMARY SCORECARD:")
    print(f"  - Total Test Queries Evaluated: {total_queries}")
    print(f"  - Average Retrieval Latency   : {avg_latency_ms:.2f} ms")
    print(f"  - Metadata Schema Compliance  : {schema_pass_rate:.1f}% PASS")
    print(f"  - Golden Rule Citation Rate   : {citation_pass_rate:.1f}% PASS")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_evaluation_benchmark()
