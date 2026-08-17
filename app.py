"""
app.py - Interactive Clinical Decision Support Dashboard (Bilingual Arabic & English Streamlit Web UI)

Modern, high-aesthetic web interface for clinicians to query 360° Clinical Diabetes Guidelines,
view AI-synthesized recommendations using OpenRouter/OpenAI LLM with inline citations,
filter by guideline documents, and inspect evidence chunks satisfying the Golden Rule.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Top-level backend imports
import ingest
from retrieve import query_clinical_rag, PERSIST_DIR
from query import generate_clinical_recommendation, is_conversational_query

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Clinical Decision Support - Diabetes Guidelines",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical dashboard styling, dark mode compatibility, and RTL support
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 1.5rem;
    }
    .badge {
        background-color: #2D3748;
        color: #E2E8F0;
        padding: 0.25rem 0.65rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .golden-rule-box {
        background-color: rgba(49, 130, 206, 0.15);
        border: 1px solid #3182CE;
        color: #63B3ED;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Cloud Automatic Ingestion Initializer (runs native Linux build if missing/corrupt)
@st.cache_resource(show_spinner=True)
def initialize_cloud_vectorstore():
    try:
        if not os.path.exists(PERSIST_DIR) or len(os.listdir(PERSIST_DIR)) == 0:
            with st.spinner("⚡ Initializing Clinical Guidelines Vector Database for Cloud Deployment..."):
                downloaded = ingest.fetch_all_guidelines(ingest.DOCS_DIR)
                sections = []
                for doc_name, path in downloaded:
                    sections.extend(ingest.parse_pdf_structure(path, doc_name))
                docs = ingest.chunk_section_data(sections)
                ingest.store_in_chromadb(docs, PERSIST_DIR)
            return True
    except Exception as e:
        print(f"Warning during cloud vectorstore init: {e}")
    return False

# Trigger cloud initialization
initialize_cloud_vectorstore()

# Read Secrets if deployed on Streamlit Cloud
if "OPENROUTER_API_KEY" in st.secrets:
    os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Sidebar Language Selector
st.sidebar.image("https://img.icons8.com/color/96/000000/doctor-female.png", width=70)
language = st.sidebar.radio("🌐 Language / اللغة:", ["🇸🇦 العربية", "🇬🇧 English"])
is_arabic = "العربية" in language

st.sidebar.title("إعدادات النظام" if is_arabic else "Clinical Settings")

# Document Selector
if is_arabic:
    DOCUMENT_OPTIONS = [
        "جميع المراجع السريرية (بحث شامل 360°)",
        "who_definition_and_diagnosis_of_diabetes.pdf",
        "diabetes_diagnosis_and_classification.pdf",
        "type2_diabetes_pharmacotherapy_management.pdf",
        "pediatric_and_adolescent_diabetes.pdf",
        "metabolic_syndrome_cardiovascular_risk.pdf",
        "thyroid_and_endocrine_comorbidities.pdf",
        "gestational_diabetes_maternal_care.pdf"
    ]
else:
    DOCUMENT_OPTIONS = [
        "All Clinical Guidelines (360° Search)",
        "who_definition_and_diagnosis_of_diabetes.pdf",
        "diabetes_diagnosis_and_classification.pdf",
        "type2_diabetes_pharmacotherapy_management.pdf",
        "pediatric_and_adolescent_diabetes.pdf",
        "metabolic_syndrome_cardiovascular_risk.pdf",
        "thyroid_and_endocrine_comorbidities.pdf",
        "gestational_diabetes_maternal_care.pdf"
    ]

doc_label = "اختيار الدليل الإرشادي:" if is_arabic else "Filter by Guideline Document:"
selected_doc = st.sidebar.selectbox(doc_label, DOCUMENT_OPTIONS)

k_label = "عدد المقاطع السريرية (Top-K):" if is_arabic else "Number of Evidence Chunks (Top-K):"
top_k_chunks = st.sidebar.slider(k_label, min_value=1, max_value=5, value=3)

# Provider status
openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
openai_key = os.getenv("OPENAI_API_KEY", "")
has_key = bool((openrouter_key or openai_key) and (openrouter_key != "your_openai_api_key_here"))

st.sidebar.markdown("---")
st.sidebar.markdown("### حالة النظام" if is_arabic else "### System Status")
st.sidebar.markdown(f"**نموذج التضمين:** `paraphrase-multilingual-MiniLM-L12-v2`" if is_arabic else "**Embedding Model:** `paraphrase-multilingual-MiniLM-L12-v2`")
st.sidebar.markdown(f"**قاعدة البيانات المتجهة:** `ChromaDB (7 Clinical Guidelines)`" if is_arabic else "**Vector Database:** `ChromaDB (7 Clinical Guidelines)`")
st.sidebar.markdown(f"**توليد الذكاء الاصطناعي:** `{'OpenRouter (openai/gpt-4o-mini)' if has_key else 'Deterministic Synthesizer'}`" if is_arabic else f"**LLM Generation:** `{'OpenRouter (openai/gpt-4o-mini)' if has_key else 'Deterministic Synthesizer'}`")

# Main Title Header & Banners
if is_arabic:
    st.markdown('<div class="main-header rtl-text">🩺 نظام دعم القرار السريري - الدلائل الإرشادية لمرض السكري</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header rtl-text">نظام استرجاع المعلومات وتوليد التوصيات السريرية الموثقة (RAG)</div>', unsafe_allow_html=True)
    st.markdown('<div class="golden-rule-box rtl-text">📌 القاعدة الذهبية: "لا ادعاء بدون مرجع". كل استجابة سريرية موثقة بدقة باسم المستند، القسم، ورقم الصفحة.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="main-header">🩺 Clinical Decision Support System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Retrieval-Augmented Generation (RAG) for Diabetes Guidelines</div>', unsafe_allow_html=True)
    st.markdown('<div class="golden-rule-box">📌 Golden Rule Enforced: "No claim without a citation." Every retrieved response is traceable back to its source document, section, and page.</div>', unsafe_allow_html=True)

# Preset clinical query buttons
if is_arabic:
    st.markdown("##### أسئلة سريرية شائعة:")
    col1, col2, col3, col4 = st.columns(4)
    preset_query = None
    if col1.button("🔍 معايير تشخيص السكري"):
        preset_query = "ما هي معايير تشخيص مرض السكري بناءً على سكر الدم الصائم والنسبة التراكمية؟"
    if col2.button("💊 خطة العلاج الدوائي"):
        preset_query = "ما هي التوصيات العلاجية والدوائية الأولى لمرضى السكري من النوع الثاني؟"
    if col3.button("👶 سكري الأطفال"):
        preset_query = "ما هي بروتوكولات ومستهدفات السكر لدى الأطفال والمراهقين؟"
    if col4.button("🤰 سكري الحمل"):
        preset_query = "ما هي معايير وبروتوكولات فحص وتشخيص سكري الحمل لدى الحوامل؟"
else:
    st.markdown("##### Sample Clinical Queries:")
    col1, col2, col3, col4 = st.columns(4)
    preset_query = None
    if col1.button("🔍 Diagnostic Criteria"):
        preset_query = "What are the fasting blood glucose criteria for diagnosing diabetes?"
    if col2.button("💊 Metformin & Pharmacotherapy"):
        preset_query = "What are the first-line medication recommendations for Type 2 diabetes?"
    if col3.button("👶 Pediatric Diabetes"):
        preset_query = "What are the glycemic targets and protocols for children and adolescents with diabetes?"
    if col4.button("🤰 Gestational Diabetes"):
        preset_query = "What are the screening protocols for gestational diabetes during pregnancy?"

# Query Input Text Area
query_label = "اكتب السؤال أو الاستفسار السريري:" if is_arabic else "Enter Clinical Question or Search Term:"
default_text = preset_query if preset_query else ("ما هي معايير تشخيص مرض السكري؟" if is_arabic else "What are the criteria for diagnosing diabetes?")

user_query = st.text_input(query_label, value=default_text)

btn_label = "🚀 استرجاع الدلائل وتوليد التوصية السريرية" if is_arabic else "🚀 Retrieve & Synthesize Recommendation"

if st.button(btn_label, type="primary"):
    if not user_query.strip():
        st.warning("يرجى كتابة سؤال سريري." if is_arabic else "Please enter a clinical question.")
    else:
        doc_filter = None if (selected_doc.startswith("All") or selected_doc.startswith("جميع")) else selected_doc

        # Check intent first
        if is_conversational_query(user_query):
            synthesis_text = generate_clinical_recommendation(user_query, top_k=top_k_chunks, doc_filter=doc_filter, lang="ar" if is_arabic else "en")
            st.markdown("### 🤖 تعريف النظام" if is_arabic else "### 🤖 System Identity")
            st.info(synthesis_text)
        else:
            spinner_msg = "جاري البحث في الدلائل الطبية وتوليد التوصية بواسطة الذكاء الاصطناعي..." if is_arabic else "Searching multi-document vector index & synthesizing evidence..."
            with st.spinner(spinner_msg):
                try:
                    results_with_scores = query_clinical_rag(user_query, top_k=top_k_chunks, doc_filter=doc_filter)
                except Exception as e:
                    print(f"ChromaDB query retry: {e}")
                    # Re-initialize native Linux build if vectorstore was corrupted
                    ingest.main()
                    results_with_scores = query_clinical_rag(user_query, top_k=top_k_chunks, doc_filter=doc_filter)

                synthesis_text = generate_clinical_recommendation(user_query, top_k=top_k_chunks, doc_filter=doc_filter, lang="ar" if is_arabic else "en")

            if not results_with_scores:
                st.warning("لم يتم العثور على مقاطع سريرية مطابقة." if is_arabic else "No relevant evidence chunks found for this query.")
            else:
                st.markdown("### 📋 التوصية السريرية الموثقة" if is_arabic else "### 📋 Synthesized Clinical Recommendation")
                
                citations_list = []
                for idx, (doc, score) in enumerate(results_with_scores, 1):
                    meta = doc.metadata
                    dname = meta.get("document_name", "UNKNOWN")
                    sec = meta.get("section_title", "N/A")
                    page = meta.get("page_number", "N/A")
                    if is_arabic:
                        citations_list.append(f"مرجع-{idx}: المستند '{dname}' | الصفحة {page} | القسم '{sec}'")
                    else:
                        citations_list.append(f"Ref-{idx}: {dname}, Page {page}, Section '{sec}'")

                # High-contrast container for synthesized text
                with st.container():
                    st.info(synthesis_text)

                # Display Evidence Chunks with Badges
                st.markdown("### 🔍 المقاطع السريرية والمراجع الدقيقة" if is_arabic else "### 🔍 Retrieved Evidence Chunks & Citations")

                for idx, (doc, score) in enumerate(results_with_scores, 1):
                    meta = doc.metadata
                    dname = meta.get("document_name", "UNKNOWN")
                    sec = meta.get("section_title", "N/A")
                    page = meta.get("page_number", "N/A")
                    chunk_id = meta.get("chunk_id", "N/A")

                    expander_label = f"مرجع #{idx}: {dname} (الصفحة {page}) - القسم: {sec}" if is_arabic else f"Reference #{idx}: {dname} (Page {page}) - Section: {sec}"

                    with st.expander(expander_label, expanded=(idx == 1)):
                        st.markdown(f"**📄 Document:** `{dname}` | **📖 Page:** `{page}` | **🔖 Section:** `{sec}`")
                        st.markdown(f"**Chunk ID:** `{chunk_id}`")
                        st.code(doc.page_content.strip(), language="text")

                # Report Download
                report_title = "تقرير دعم القرار السريري - Clinical Decision Support Report"
                report_content = f"{report_title}\nQuery/السؤال: {user_query}\n\n{synthesis_text}\n\nالمراجع والدلائل السريرية / Evidence Sources:\n" + "\n".join([f"- {c}" for c in citations_list])
                
                st.download_button(
                    label="📥 تحميل التقرير السريري (TXT)" if is_arabic else "📥 Download Clinical Report",
                    data=report_content,
                    file_name="clinical_decision_report.txt",
                    mime="text/plain"
                )

st.markdown("---")
footer_text = "نظام دعم القرار السريري لمرض السكري | مجهّز بواسطة OpenRouter (GPT-4o-Mini), LangChain & ChromaDB" if is_arabic else "Clinical Decision Support System RAG Pipeline | Powered by OpenRouter (GPT-4o-Mini), LangChain & ChromaDB"
st.markdown(f"<div style='text-align: center; color: #718096; font-size: 0.9rem;'>{footer_text}</div>", unsafe_allow_html=True)
