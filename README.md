# Clinical Decision Support System - Diabetes Guidelines RAG Pipeline

A production-grade, structure-aware **Retrieval-Augmented Generation (RAG)** pipeline designed for Clinical Decision Support using Diabetes Guidelines.

Built with **LangChain**, **PyMuPDF (`fitz`)**, **ChromaDB**, and configurable embeddings (**OpenAI** or free local **HuggingFace Sentence-Transformers**).

---

## 🌟 Key Features

1. **The Golden Rule**: *"No claim without a citation."* Every retrieved chunk contains full provenance back to its exact section title, page number, document name, and unique chunk ID.
2. **Mandatory Metadata Schema**: Strictly enforces metadata fields:
   - `document_name`
   - `section_title`
   - `page_number`
   - `chunk_id`
3. **Structure-Aware Parsing**: Uses PyMuPDF (`fitz`) layout analysis to extract headers, sections, page numbers, and clinical tables without naive text splitting.
4. **Section-Aware Chunking**: Target windowing of 300–500 tokens with 10–15% token overlap within document sections.
5. **Zero-Setup Local Vector DB**: Embedded lightweight local ChromaDB (`./chroma_db`).
6. **Auto-Fetching**: Automated HTTP fetching of open-access WHO Diabetes Guidelines PDF.

---

## 🚀 Quickstart Guide ("Zero to Hero")

### Step 1: Clone or Navigate to Project Directory
```bash
cd c:\Users\abdul\Downloads\hackathon
```

### Step 2: Install Dependencies
Install all required dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env` (already pre-configured with local HuggingFace embeddings for free, zero-cost execution):
```bash
# Optional: Set your OpenAI API key if EMBEDDING_PROVIDER=openai
# OPENAI_API_KEY=sk-...
```

---

## 📥 Ingestion Pipeline (`ingest.py`)

Run the automated ingestion script:
```bash
python ingest.py
```

### What `ingest.py` Does:
1. **Auto-Fetch**: Downloads the WHO Clinical Diabetes Guideline PDF if not present locally as `diabetes_guidelines.pdf`.
2. **Structure Parsing**: Parses layout elements, font sizes, headings, and page boundaries using PyMuPDF.
3. **Section Chunking**: Splits text per section into 300–500 token chunks with overlap.
4. **Metadata Tagging**: Attaches `document_name`, `section_title`, `page_number`, and `chunk_id`.
5. **Vector Ingestion**: Embeds and indexes all chunks into local `./chroma_db`.

---

## 🔍 Retrieval Pipeline (`retrieve.py`)

Run the clinical query retrieval tool:
```bash
python retrieve.py
```

### Run Custom Clinical Queries
Pass your custom clinical question as a command-line argument:
```bash
python retrieve.py "What are the first-line medication recommendations for Type 2 diabetes?"
```

```bash
python retrieve.py "What are the fasting blood glucose thresholds for diagnosing diabetes?"
```

---

## 📁 Project Structure

```
├── .env                  # Active environment variables
├── .env.example          # Template environment file
├── requirements.txt      # Python dependencies
├── ingest.py             # Auto-download, parse, chunk, metadata & ChromaDB ingestion
├── retrieve.py           # ChromaDB search & explicit citation printout
├── README.md             # Project documentation
└── chroma_db/            # Generated local ChromaDB vector store (after ingest)
```

---

## 🔒 Citation Compliance (The Golden Rule)

Example terminal output from `retrieve.py`:

```text
================================================================================
                  CLINICAL RAG RETRIEVAL QUERY RESULTS                  
================================================================================
QUERY: "What are the criteria for diagnosing diabetes?"
RETRACTING TOP 3 MOST RELEVANT CLINICAL GUIDELINE CHUNKS...

--- [RESULT #1] (Relevance Score: 0.8412) ---
📌 CITATION (Golden Rule):
   [Source Document: diabetes_guidelines.pdf | Section: 'Diagnosis and screening' | Page: 3 | Chunk ID: diabetes_guidelines.pdf_p3_c5]
📄 METADATA DETAILS:
   - Document Name : diabetes_guidelines.pdf
   - Section Title : Diagnosis and screening
   - Page Number   : 3
   - Chunk ID      : diabetes_guidelines.pdf_p3_c5
📝 RETRIEVED CONTENT CHUNK:
   --------------------------------------------------------------------------
   Fasting plasma glucose >= 7.0 mmol/L (126 mg/dL) or 2-hour post-load glucose...
   --------------------------------------------------------------------------
```
