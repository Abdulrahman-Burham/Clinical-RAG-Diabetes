# 🩺 360-Degree Clinical Decision Support System - Diabetes Guidelines RAG Pipeline

A production-grade, multi-document **Retrieval-Augmented Generation (RAG)** pipeline designed for Clinical Decision Support using 6 comprehensive Diabetes and Endocrine Guidelines documents.

Built with **LangChain**, **PyMuPDF (`pymupdf`)**, **ChromaDB**, **Streamlit**, and configurable embeddings (**OpenAI** or free local **HuggingFace Sentence-Transformers**).

---

## 🌟 Key Architectural Principles

1. **The Golden Rule**: *"No claim without a citation."* Every retrieved chunk and generated recommendation contains full provenance back to its exact document name, section title, page number, and unique chunk ID.
2. **Mandatory Metadata Schema**: Strictly enforces metadata fields across all chunks:
   - `document_name`
   - `section_title`
   - `page_number`
   - `chunk_id`
3. **Structure-Aware Multi-PDF Parsing**: Uses PyMuPDF layout analysis to extract headers, sections, page numbers, and clinical tables across 6 clinical guidelines documents without naive text splitting.
4. **Section-Aware Chunking**: Target windowing of 300–500 tokens with 10–15% token overlap within document sections.
5. **Zero-Setup Local Vector DB**: Embedded lightweight local ChromaDB (`./chroma_db`) with over 1,000 indexed clinical chunks.
6. **Clinical RAG Generation & Evaluation**:
   - `query.py`: Synthesizes evidence chunks into clinical recommendations with inline citations `[Doc | Sec | Page]`.
   - `eval.py`: Automated benchmarking suite testing retrieval latency, metadata schema compliance, and citation coverage.
   - `app.py`: Interactive Streamlit Web UI Dashboard for doctors.

---

## 📚 Ingested Medical Guidelines Library (360° Coverage)

The pipeline automatically fetches and indexes **6 open-access clinical guideline PDFs** inside `./guidelines_docs/`:

1. `diabetes_diagnosis_and_classification.pdf`: Diagnostic criteria, fasting blood glucose, A1C thresholds, and classification (Types 1, 2, MODY).
2. `type2_diabetes_pharmacotherapy_management.pdf`: Adult Type 2 diabetes management, Metformin, SGLT2 inhibitors, GLP-1 RA, and insulin protocols.
3. `pediatric_and_adolescent_diabetes.pdf`: Diagnosis, glycemic targets, and treatment protocols for children and adolescents.
4. `metabolic_syndrome_cardiovascular_risk.pdf`: Cardiovascular risk factor management, dyslipidemia, and hypertension in diabetes.
5. `thyroid_and_endocrine_comorbidities.pdf`: Management of co-existing endocrine and autoimmune conditions.
6. `gestational_diabetes_maternal_care.pdf`: Screening, diagnostic criteria, and management of Gestational Diabetes Mellitus (GDM) during pregnancy.

---

## 🚀 Quickstart Guide ("Zero to Hero")

### Step 1: Navigate to Project Directory
```bash
cd c:\Users\abdul\Downloads\hackathon
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env` (pre-configured for free local HuggingFace embeddings):
```bash
# Optional: Set your OpenAI API key if EMBEDDING_PROVIDER=openai
# OPENAI_API_KEY=sk-...
```

---

## 📥 1. Multi-Document Ingestion (`ingest.py`)

Auto-fetch all 6 guideline PDFs, perform structure-aware parsing, section chunking, metadata tagging, and vector storage in `./chroma_db`:

```bash
python ingest.py
```

---

## 🔍 2. Traceable Vector Retrieval (`retrieve.py`)

Run multi-document vector retrieval with exact Golden Rule citations:

```bash
python retrieve.py "What are the fasting blood glucose criteria for diagnosing diabetes?"
```

### Filter by Specific Guideline Document:
```bash
python retrieve.py "What are the glycemic targets?" --doc pediatric_and_adolescent_diabetes.pdf
```

---

## 🤖 3. Generative Clinical Engine (`query.py`)

Generate a synthesized clinical recommendation with inline citations:

```bash
python query.py "What are the first-line medication recommendations for Type 2 diabetes?"
```

---

## 📊 4. Run RAG Evaluation Benchmark (`eval.py`)

Run the clinical benchmark suite testing latency and citation pass rates:

```bash
python eval.py
```

---

## 💻 5. Launch Interactive Web UI (`app.py`)

Launch the Streamlit Clinical Decision Support Dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to use the interactive dashboard!

---

## 📁 Complete Project Structure

```
├── .env                                  # Active environment variables
├── .env.example                          # Template environment file
├── requirements.txt                      # Python dependencies
├── ingest.py                             # Multi-PDF downloader, structure parser, section chunker & ChromaDB indexer
├── retrieve.py                           # Vector search & Golden Rule citation printer (with document filtering)
├── query.py                              # RAG Clinical Recommendation Generator with inline citations
├── eval.py                               # RAG Benchmark harness testing latency, schema, & citation rates
├── app.py                                # Interactive Streamlit Web UI Dashboard
├── README.md                             # Project documentation
├── guidelines_docs/                      # Downloaded Clinical Guideline PDFs (6 documents)
└── chroma_db/                            # Local ChromaDB vector database (1000+ indexed chunks)
```
