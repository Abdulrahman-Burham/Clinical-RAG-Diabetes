"""
ingest.py - Multi-Document Clinical Guidelines Data Ingestion & Structure-Aware RAG Pipeline

Steps:
1. Auto-fetch 6 open-access Clinical Diabetes Guidelines PDFs into ./guidelines_docs/.
2. Structure-aware PDF parsing using PyMuPDF (pymupdf) to extract layout text, page numbers, and headings.
3. Section-aware chunking (300-500 tokens, 10-15% overlap) + Noise & Bibliography Filtering.
4. Strict Metadata tagging: document_name, section_title, page_number, chunk_id.
5. Multilingual Vector embedding (Arabic & English cross-lingual RAG) and indexing into local ChromaDB folder (./chroma_db).
"""

import os
import re
import shutil
import requests
import pymupdf  # PyMuPDF
from dotenv import load_dotenv

from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
DOCS_DIR = "./guidelines_docs"
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = "diabetes_guidelines"

# Comprehensive 360° Clinical Diabetes Guidelines Catalog
GUIDELINES_CATALOG = [
    {
        "filename": "diabetes_diagnosis_and_classification.pdf",
        "title": "Diagnosis & Classification of Diabetes Mellitus",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK585641/pdf/Bookshelf_NBK585641.pdf"
    },
    {
        "filename": "type2_diabetes_pharmacotherapy_management.pdf",
        "title": "Adult Type 2 Diabetes Management & Pharmacotherapy",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK585639/pdf/Bookshelf_NBK585639.pdf"
    },
    {
        "filename": "pediatric_and_adolescent_diabetes.pdf",
        "title": "Pediatric & Adolescent Diabetes Management",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK585640/pdf/Bookshelf_NBK585640.pdf"
    },
    {
        "filename": "metabolic_syndrome_cardiovascular_risk.pdf",
        "title": "Metabolic Syndrome & Cardiovascular Risk Management",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK585642/pdf/Bookshelf_NBK585642.pdf"
    },
    {
        "filename": "thyroid_and_endocrine_comorbidities.pdf",
        "title": "Thyroid & Endocrine Co-morbidities in Diabetes",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK585643/pdf/Bookshelf_NBK585643.pdf"
    },
    {
        "filename": "gestational_diabetes_maternal_care.pdf",
        "title": "Gestational Diabetes & Maternal Care Guidelines",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK585644/pdf/Bookshelf_NBK585644.pdf"
    }
]

def fetch_all_guidelines(docs_dir: str):
    """Step 1: Automatically fetch all clinical guideline PDFs into local docs directory."""
    os.makedirs(docs_dir, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    downloaded_files = []

    print(f"\n[Step 1] Verifying & Fetching {len(GUIDELINES_CATALOG)} Clinical Guidelines PDFs...")
    for idx, item in enumerate(GUIDELINES_CATALOG, 1):
        filename = item["filename"]
        title = item["title"]
        url = item["url"]
        target_path = os.path.join(docs_dir, filename)

        if os.path.exists(target_path):
            with open(target_path, "rb") as f:
                header = f.read(10)
            if header.startswith(b"%PDF"):
                print(f"  [{idx}/{len(GUIDELINES_CATALOG)}] '{filename}' already exists locally ({title}).")
                downloaded_files.append((filename, target_path))
                continue

        print(f"  [{idx}/{len(GUIDELINES_CATALOG)}] Downloading '{filename}' ({title})...")
        try:
            r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            r.raise_for_status()
            if r.content.startswith(b"%PDF"):
                with open(target_path, "wb") as f:
                    f.write(r.content)
                print(f"      -> Successfully saved '{filename}' ({len(r.content)} bytes).")
                downloaded_files.append((filename, target_path))
            else:
                print(f"      -> Warning: Response from {url} was not a valid PDF header.")
        except Exception as e:
            print(f"      -> Error downloading {filename}: {e}")

    print(f"[Step 1] Completed fetch. Total active PDFs: {len(downloaded_files)}/{len(GUIDELINES_CATALOG)}")
    return downloaded_files

def is_noise_or_reference_block(text: str, section_title: str = "") -> bool:
    """
    Strictly filters out author affiliations, publisher metadata, headers,
    and Bibliography/References sections.
    """
    t_clean = text.strip()
    if len(t_clean) < 60:
        return True

    # 1. Ignore References/Bibliography sections completely
    sec_lower = section_title.lower()
    if any(ref_word in sec_lower for ref_word in ["reference", "bibliography", "literature cited", "endnotes"]):
        return True

    # 2. Check for bibliography citation patterns in text
    # e.g., "1. Smith J, et al. Diabetes Care 2019;42:123-130." or "Pak J Med Sci"
    citation_patterns = [
        r'\b(?:10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b',  # DOIs
        r'\b(?:Pak J Med Sci|Diabetes Care|Endocrinol Metab|Ups J Med Sci|Med Sci|Arch Endocrinol)\b',
        r'^\s*\d{1,3}\.\s+[A-Z][a-z]+(?:\s+[A-Z]{1,3})?,',  # Bibliographic numbering "43. Sharma B,"
        r'Available at https?://',
        r'Accessed\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}'
    ]
    for pat in citation_patterns:
        if re.search(pat, t_clean):
            return True

    # 3. Common author/header noise patterns
    noise_patterns = [
        r"Gabriela\s+Brenta",
        r"Cesar\s+Milstein\s+Hospital",
        r"Buenos\s+Aires",
        r"The\s+Author\(s\)\s+20\d\d",
        r"Warning:\s+The\s+NCBI\s+web\s+site",
        r"An\s+official\s+website\s+of\s+the\s+United\s+States",
        r"Search\s+databaseBooksAll",
        r"Browse\s+Title"
    ]
    for pat in noise_patterns:
        if re.search(pat, t_clean, re.IGNORECASE):
            return True

    return False

def parse_pdf_structure(pdf_path: str, doc_name: str):
    """
    Step 2: Structure-Aware Parsing using PyMuPDF (pymupdf).
    Extracts text blocks page-by-page while detecting headings/sections and page numbers.
    Filters out Bibliography / References sections.
    """
    doc = pymupdf.open(pdf_path)
    sections_data = []
    current_section = f"Clinical Guidelines ({doc_name})"

    # Regex heuristic for detecting section titles
    section_title_pattern = re.compile(
        r'^(?:[0-9]{1,2}\.|\b(?:SECTION|CHAPTER|PROTOCOL|GUIDELINE|PART|STEP|ANNEX|APPENDIX|DIAGNOSIS|MANAGEMENT|TREATMENT|CRITERIA|SCREENING|MEDICATION|MONITORING|COMPLICATIONS|PREVENTION|OVERVIEW|SUMMARY)\b)',
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

                    # Structure-Aware Heading Detection
                    if (avg_font_size >= 12.5 or (is_bold and len(line_str) < 80)) and len(line_str) > 3:
                        if section_title_pattern.search(line_str) or is_bold or avg_font_size >= 12.5:
                            if not is_noise_or_reference_block(line_str, line_str):
                                current_section = line_str

                    block_lines.append(line_str)

                if block_lines:
                    block_content = "\n".join(block_lines)
                    # Filter out non-clinical noise and references
                    if not is_noise_or_reference_block(block_content, current_section):
                        page_text_blocks.append({
                            "text": block_content,
                            "section_title": current_section,
                            "page_number": page_num,
                            "document_name": doc_name
                        })

        for b in page_text_blocks:
            sections_data.append(b)

    doc.close()
    return sections_data

def chunk_section_data(all_sections_data):
    """
    Step 3 & 4: Section-Aware Chunking & Strict Metadata Tagging.
    Target chunk size: ~300-500 tokens (approx 1200-1800 chars) with 10-15% overlap (~200 chars).
    Attaches mandatory metadata fields: document_name, section_title, page_number, chunk_id.
    """
    print(f"[Step 3 & 4] Applying Section-Aware Chunking (300-500 tokens, 10-15% overlap)...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", " ", ""]
    )

    documents = []
    chunk_counter = 0

    for item in all_sections_data:
        text = item["text"]
        section_title = item["section_title"]
        page_number = item["page_number"]
        document_name = item["document_name"]

        if not text.strip() or is_noise_or_reference_block(text, section_title):
            continue

        raw_chunks = text_splitter.split_text(text)

        for chunk_text in raw_chunks:
            if is_noise_or_reference_block(chunk_text, section_title):
                continue
                
            chunk_counter += 1
            chunk_id = f"{document_name}_p{page_number}_c{chunk_counter}"

            # Strict Metadata Schema Validation
            metadata = {
                "document_name": str(document_name),
                "section_title": str(section_title),
                "page_number": int(page_number),
                "chunk_id": str(chunk_id)
            }

            doc = Document(page_content=chunk_text, metadata=metadata)
            documents.append(doc)

    print(f"[Step 3 & 4] Created {len(documents)} clean clinical body chunks across all guideline documents.")
    return documents

def get_embedding_function():
    """Initializes configurable embedding provider (Multilingual HuggingFace or OpenAI)."""
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if provider == "openai" and openai_key and openai_key != "your_openai_api_key_here":
        print(f"Using OpenAI Embeddings ({model_name or 'text-embedding-3-small'})...")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name or "text-embedding-3-small",
            openai_api_key=openai_key
        )
    else:
        print(f"Using Multilingual HuggingFace Embeddings ({model_name})...")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)

def store_in_chromadb(documents, persist_dir: str):
    """Step 5: Store chunks and vector embeddings into local ChromaDB."""
    print(f"[Step 5] Cleaning old database directory '{persist_dir}' for fresh ingestion...")
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning clearing persist dir: {e}")

    print(f"[Step 5] Initializing local ChromaDB at directory: '{persist_dir}'...")
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    embedding_function = get_embedding_function()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME
    )
    
    print(f"[Step 5] Ingestion Complete! Indexed {len(documents)} clean clinical body chunks into ChromaDB at '{persist_dir}'.")
    return vectorstore

def main():
    print("=" * 75)
    print("  CLINICAL DECISION SUPPORT SYSTEM - MULTILINGUAL DATA INGESTION PIPELINE  ")
    print("=" * 75)

    # Step 1: Auto-fetch all PDFs
    downloaded_files = fetch_all_guidelines(DOCS_DIR)

    # Step 2: Parse structure across all PDFs
    all_sections_data = []
    print(f"\n[Step 2] Performing Structure-Aware parsing across {len(downloaded_files)} PDFs...")
    for doc_name, pdf_path in downloaded_files:
        sections = parse_pdf_structure(pdf_path, doc_name)
        print(f"  -> Extracted {len(sections)} clean body blocks from '{doc_name}'.")
        all_sections_data.extend(sections)

    # Step 3 & 4: Chunk & Tag Strict Metadata Schema
    documents = chunk_section_data(all_sections_data)

    # Validate Strict Metadata Schema before storing
    required_keys = {"document_name", "section_title", "page_number", "chunk_id"}
    for doc in documents:
        meta_keys = set(doc.metadata.keys())
        if not required_keys.issubset(meta_keys):
            raise ValueError(f"Metadata missing required schema keys! Found keys: {meta_keys}")

    # Step 5: Index in ChromaDB
    store_in_chromadb(documents, PERSIST_DIR)

    print("\n[SUCCESS] Multilingual ingestion pipeline executed successfully!")

if __name__ == "__main__":
    main()
