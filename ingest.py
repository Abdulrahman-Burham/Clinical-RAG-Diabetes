"""
ingest.py - Data Ingestion & Structure-Aware RAG Retrieval Pipeline for Diabetes Guidelines

Steps:
1. Auto-fetch open-access Clinical Guidelines PDF via HTTP GET (requests) with robust validation & fallback URLs.
2. Structure-aware PDF parsing using PyMuPDF (pymupdf) to extract text, page numbers, and structural headings.
3. Section-aware chunking (300-500 tokens, 10-15% overlap).
4. Strict Metadata tagging: document_name, section_title, page_number, chunk_id.
5. Vector embedding and storage in local ChromaDB folder (./chroma_db).
"""

import os
import re
import requests
import pymupdf  # PyMuPDF
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
PDF_FILENAME = "diabetes_guidelines.pdf"
PRIMARY_URL = os.getenv(
    "GUIDELINE_PDF_URL",
    "https://www.ncbi.nlm.nih.gov/books/NBK585641/pdf/Bookshelf_NBK585641.pdf"
)
FALLBACK_URLS = [
    "https://www.ncbi.nlm.nih.gov/books/NBK585641/pdf/Bookshelf_NBK585641.pdf",
    "https://iris.who.int/bitstream/handle/10665/377626/Diabetes-management-protocol-eng.pdf"
]
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = "diabetes_guidelines"

def fetch_pdf(url: str, output_filename: str) -> str:
    """Step 1: Automatically fetch open-access Clinical Diabetes Guideline PDF."""
    if os.path.exists(output_filename):
        with open(output_filename, "rb") as f:
            content_start = f.read(10)
        if content_start.startswith(b"%PDF"):
            print(f"[Step 1] Valid PDF '{output_filename}' already exists locally. Skipping download.")
            return output_filename
        else:
            print(f"[Step 1] Existing '{output_filename}' is invalid. Re-downloading...")

    candidate_urls = [url] + [u for u in FALLBACK_URLS if u != url]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for idx, cand_url in enumerate(candidate_urls, 1):
        print(f"[Step 1] Attempting to download PDF (Attempt {idx}) from: {cand_url}")
        try:
            response = requests.get(cand_url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            content = response.content

            if content.startswith(b"%PDF"):
                with open(output_filename, "wb") as f:
                    f.write(content)
                print(f"[Step 1] Successfully downloaded and verified '{output_filename}' ({len(content)} bytes).")
                return output_filename
            else:
                print(f"[Step 1] Response from {cand_url} was not a valid PDF binary. Trying next candidate...")
        except Exception as e:
            print(f"[Step 1] Download attempt from {cand_url} failed: {e}")

    raise RuntimeError("Failed to fetch a valid PDF file from all candidate URLs.")

def parse_pdf_structure(pdf_path: str):
    """
    Step 2: Structure-Aware Parsing using PyMuPDF (fitz/pymupdf).
    Extracts text blocks page-by-page while detecting headings/sections and page numbers.
    Returns a list of section blocks with metadata.
    """
    print(f"[Step 2] Performing Structure-Aware parsing on '{pdf_path}'...")
    doc = pymupdf.open(pdf_path)
    print(f"Total pages detected: {len(doc)}")

    sections_data = []
    current_section = "Clinical Overview & Introduction"

    # Regex heuristic for detecting section titles
    section_title_pattern = re.compile(
        r'^(?:[0-9]{1,2}\.|\b(?:SECTION|CHAPTER|PROTOCOL|GUIDELINE|PART|STEP|ANNEX|APPENDIX|DIAGNOSIS|MANAGEMENT|TREATMENT|CRITERIA|SCREENING|MEDICATION|MONITORING|COMPLICATIONS|PREVENTION|OVERVIEW|SUMMARY|APPENDIX)\b)',
        re.IGNORECASE
    )

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1  # 1-based index
        page_dict = page.get_text("dict")
        page_text_blocks = []

        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_lines = []
                for line in block.get("lines", []):
                    line_text = ""
                    font_sizes = []
                    is_bold = False

                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if span_text:
                            line_text += span_text + " "
                            font_sizes.append(span.get("size", 10))
                            if "bold" in span.get("font", "").lower() or span.get("flags", 0) & 2:
                                is_bold = True

                    line_str = line_text.strip()
                    if not line_str:
                        continue

                    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10

                    # Structure-Aware Heading Detection:
                    # Font size threshold (>12pt or bold short line or matches regex pattern)
                    if (avg_font_size >= 12.5 or (is_bold and len(line_str) < 80)) and len(line_str) > 3:
                        if section_title_pattern.search(line_str) or is_bold or avg_font_size >= 12.5:
                            # Update active section title
                            current_section = line_str

                    block_lines.append(line_str)

                if block_lines:
                    block_content = "\n".join(block_lines)
                    page_text_blocks.append({
                        "text": block_content,
                        "section_title": current_section,
                        "page_number": page_num
                    })

        # Group page text blocks under their section
        for b in page_text_blocks:
            sections_data.append(b)

    doc.close()
    print(f"[Step 2] Finished structure-aware parsing. Extracted {len(sections_data)} structured text blocks.")
    return sections_data

def chunk_section_data(sections_data, document_name: str):
    """
    Step 3 & 4: Section-Aware Chunking & Strict Metadata Tagging.
    Target chunk size: ~300-500 tokens (approx 1200-2000 chars) with 10-15% overlap (~150-250 chars).
    Attaches mandatory metadata fields: document_name, section_title, page_number, chunk_id.
    """
    print(f"[Step 3 & 4] Applying Section-Aware Chunking (300-500 tokens, 10-15% overlap)...")
    
    # Token-equivalent character splitter: ~1400 chars target (~350 tokens), ~200 overlap (~50 tokens, ~14%)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", " ", ""]
    )

    documents = []
    chunk_counter = 0

    for item in sections_data:
        text = item["text"]
        section_title = item["section_title"]
        page_number = item["page_number"]

        if not text.strip():
            continue

        raw_chunks = text_splitter.split_text(text)

        for sub_idx, chunk_text in enumerate(raw_chunks):
            chunk_counter += 1
            chunk_id = f"{document_name}_p{page_number}_c{chunk_counter}"

            # Strict Metadata Schema Validation:
            # Must include exactly document_name, section_title, page_number, chunk_id
            metadata = {
                "document_name": str(document_name),
                "section_title": str(section_title),
                "page_number": int(page_number),
                "chunk_id": str(chunk_id)
            }

            doc = Document(page_content=chunk_text, metadata=metadata)
            documents.append(doc)

    print(f"[Step 3 & 4] Created {len(documents)} section-aware chunks with strict metadata schema.")
    return documents

def get_embedding_function():
    """Initializes configurable embedding provider (OpenAI or HuggingFace fallback)."""
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    model_name = os.getenv("EMBEDDING_MODEL", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if provider == "openai" and openai_key and openai_key != "your_openai_api_key_here":
        print(f"Using OpenAI Embeddings ({model_name or 'text-embedding-3-small'})...")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name or "text-embedding-3-small",
            openai_api_key=openai_key
        )
    else:
        if provider == "openai":
            print("Notice: OPENAI_API_KEY not set. Falling back to local HuggingFace Embeddings.")
        else:
            print("Using HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def store_in_chromadb(documents, persist_dir: str):
    """Step 5: Store chunks and vector embeddings into local ChromaDB."""
    print(f"[Step 5] Initializing local ChromaDB at directory: '{persist_dir}'...")
    from langchain_community.vectorstores import Chroma

    embedding_function = get_embedding_function()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME
    )
    
    print(f"[Step 5] Ingestion Complete! Successfully indexed {len(documents)} chunks into ChromaDB at '{persist_dir}'.")
    return vectorstore

def main():
    print("=" * 70)
    print("      CLINICAL DECISION SUPPORT SYSTEM - DATA INGESTION PIPELINE      ")
    print("=" * 70)

    # Step 1: Auto-fetch PDF
    url = os.getenv("GUIDELINE_PDF_URL", PRIMARY_URL)
    pdf_path = fetch_pdf(url, PDF_FILENAME)

    # Step 2: Structure-Aware Parsing
    sections_data = parse_pdf_structure(pdf_path)

    # Step 3 & 4: Section-Aware Chunking & Strict Metadata Tagging
    documents = chunk_section_data(sections_data, PDF_FILENAME)

    # Validate Strict Metadata Schema before storing
    required_keys = {"document_name", "section_title", "page_number", "chunk_id"}
    for doc in documents:
        meta_keys = set(doc.metadata.keys())
        if not required_keys.issubset(meta_keys):
            raise ValueError(f"Metadata missing required schema keys! Found keys: {meta_keys}")

    # Step 5: Store in ChromaDB
    store_in_chromadb(documents, PERSIST_DIR)

    print("\n[SUCCESS] Ingestion pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
