"""
query.py - Generative Clinical Decision Support RAG Engine with Strict Intent & Fallback Guardrails

Retrieves relevant clinical evidence chunks from ChromaDB and synthesizes an evidence-based
clinical recommendation using OpenRouter / OpenAI LLM with inline citations [Doc | Section | Page]
enforcing the Golden Rule and strict "No hallucination / Out-of-Domain" guardrails.
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

from retrieve import query_clinical_rag

load_dotenv()

# Conversational / System identity intent keywords
CONVERSATIONAL_PATTERNS = [
    r"^\s*(?:مين\s+انت|انت\s+مين|من\s+انت|من\s+أنت|ماهو\s+هذا\s+النظام|عرف\s+نفسك|أنت\s+مين)\s*$",
    r"^\s*(?:who\s+are\s+you|what\s+are\s+you|hello|hi|hey|about\s+you)\s*$"
]

def is_conversational_query(text: str) -> bool:
    """Detects whether user query is a simple greeting or identity question."""
    t_clean = text.strip().lower()
    for pat in CONVERSATIONAL_PATTERNS:
        if re.search(pat, t_clean, re.IGNORECASE):
            return True
    return False

def generate_clinical_recommendation(query_text: str, top_k: int = 3, doc_filter: str = None, lang: str = "en"):
    """
    Generates a synthesized clinical recommendation backed by inline citations.
    Enforces strict RAG Guardrails for conversational and out-of-domain queries.
    """
    is_ar = (lang == "ar" or any("\u0600" <= c <= "\u06FF" for c in query_text))

    print("\n" + "=" * 80)
    title = "      محرك التوليد ودعم القرار السريري (RAG CLINICAL ENGINE)      " if is_ar else "      GENERATIVE CLINICAL DECISION SUPPORT ENGINE (RAG GENERATOR)      "
    print(title)
    print("=" * 80)
    print(f"QUESTION / السؤال: \"{query_text}\"")

    # Guardrail 1: Conversational / Identity Query Handling
    if is_conversational_query(query_text):
        if is_ar:
            answer = "أنا نظام دعم القرار السريري المتخصص في الدلائل الإرشادية لمرض السكري. يمكنك استفساري عن معايير التشخيص، بروتوكولات العلاج الدوائي، سكري الأطفال، وسكري الحمل."
        else:
            answer = "I am an AI-powered Clinical Decision Support System focused on Clinical Diabetes Guidelines. You can ask me about diagnostic criteria, pharmacotherapy protocols, pediatric diabetes, or gestational diabetes."
        
        print(f"\n[Intent Detection] System Identity Query Detected.\nAnswer: {answer}\n")
        return answer

    # Step 1: Retrieve top evidence chunks
    results_with_scores = query_clinical_rag(query_text, top_k=top_k, doc_filter=doc_filter)

    if not results_with_scores:
        if is_ar:
            return "عذرًا، لم يتم العثور على دلائل سريرية مطابقة في المراجع الطبية المتاحة لهذا الاستفسار."
        else:
            return "No relevant clinical evidence found in the available guidelines for this query."

    # Step 2: Build context block with strict citations
    context_blocks = []
    citations_list = []

    for idx, (doc, score) in enumerate(results_with_scores, 1):
        meta = doc.metadata
        doc_name = meta.get("document_name", "UNKNOWN")
        section = meta.get("section_title", "N/A")
        page_num = meta.get("page_number", "N/A")

        if is_ar:
            citation_tag = f"مرجع-{idx}: المستند '{doc_name}' | الصفحة {page_num} | القسم '{section}'"
        else:
            citation_tag = f"Ref-{idx}: {doc_name}, Page {page_num}, Section '{section}'"

        citations_list.append(citation_tag)
        context_blocks.append(f"[{citation_tag}]\n{doc.page_content.strip()}")

    full_context = "\n\n".join(context_blocks)

    # Step 3: Check API Key and Base URL (OpenRouter or OpenAI)
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    api_key = openrouter_key or openai_key
    api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    model_name = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

    if api_key and api_key != "your_openai_api_key_here":
        print(f"\n[*] Synthesizing clinical recommendation using OpenRouter/OpenAI LLM ({model_name})...")
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import PromptTemplate

            llm = ChatOpenAI(
                model=model_name,
                temperature=0.1,
                openai_api_key=api_key,
                openai_api_base=api_base
            )

            if is_ar:
                prompt_template = PromptTemplate.from_template(
                    """أنت طبيب استشاري ومساعد سريري خبير.
قم بالإجابة على سؤال الطبيب باللغة العربية بأسلوب سريري دقيق وموجز بناءً على سياق الدلائل الطبية المرفقة فقط.
إذا كان السؤال خارج نطاق الدلائل الطبية المرفقة أو لم يذكر السياق الإجابة، صرح بوضوح بأن المعلومة غير متوفرة في الدلائل المتاحة.
يجب توثيق كل جملة أو توصية تقوم بكتابتها بمرجع داخلي مثل [مرجع-1] أو [مرجع-2].

سياق الدلائل الطبية المقتطفة:
{context}

سؤال الطبيب: {question}

التوصية السريرية الموثقة (باللغة العربية مع توثيق المراجع):"""
                )
            else:
                prompt_template = PromptTemplate.from_template(
                    """You are an expert Clinical Decision Support Assistant.
Answer the clinician's question using ONLY the provided evidence context below.
If the context does not contain sufficient clinical information to answer the question, state: "The requested information is not available in the provided clinical guidelines."
Every single claim or recommendation you make MUST end with an inline reference tag like [Ref-1], [Ref-2], etc.

Evidence Context:
{context}

Clinician Question: {question}

Synthesized Clinical Answer (with Inline References):"""
                )

            chain = prompt_template | llm
            response = chain.invoke({"context": full_context, "question": query_text})
            answer = response.content

        except Exception as e:
            print(f"[Notice] LLM API error ({e}). Using Deterministic Clinical Evidence Synthesizer.")
            answer = build_rule_based_synthesis(query_text, results_with_scores, is_ar=is_ar)
    else:
        print("\n[*] Notice: API Key not set. Using Deterministic Clinical Evidence Synthesizer...")
        answer = build_rule_based_synthesis(query_text, results_with_scores, is_ar=is_ar)

    print("\n" + "=" * 80)
    sec_title = "                      التوصية السريرية الموثقة                      " if is_ar else "                      SYNTHESIZED CLINICAL RECOMMENDATION                      "
    print(sec_title)
    print("=" * 80)
    print(answer)
    print("\n" + "=" * 80)
    print("CITATIONS & EVIDENCE SOURCES (Golden Rule Traceability):")
    for cit in citations_list:
        print(f"  - {cit}")
    print("=" * 80 + "\n")
    return answer

def build_rule_based_synthesis(query_text: str, results_with_scores, is_ar: bool = False):
    """Fallback deterministic evidence synthesizer supporting Arabic and English."""
    if is_ar:
        lines = [f"بناءً على الدلائل الإرشادية السريرية لـ '{query_text}':\n"]
        for idx, (doc, score) in enumerate(results_with_scores, 1):
            meta = doc.metadata
            ref_tag = f"[مرجع-{idx}: {meta.get('document_name')} | الصفحة {meta.get('page_number')} | القسم '{meta.get('section_title')}']"
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 250:
                snippet = snippet[:250] + "..."
            lines.append(f"{idx}. {snippet} {ref_tag}")
        lines.append("\nالخلاصة: تنصح الدلائل الطبية أعلاه بمراجعة البروتوكولات السريرية المحددة والتحاليل الخاصة بكل مريض.")
    else:
        lines = [f"Based on retrieved Clinical Guidelines for '{query_text}':\n"]
        for idx, (doc, score) in enumerate(results_with_scores, 1):
            meta = doc.metadata
            ref_tag = f"[Ref-{idx}: {meta.get('document_name')}, Sec '{meta.get('section_title')}', Page {meta.get('page_number')}]"
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 250:
                snippet = snippet[:250] + "..."
            lines.append(f"{idx}. {snippet} {ref_tag}")
        lines.append("\nConclusion: The above guidelines recommend reviewing specific clinical protocols and patient-specific metrics.")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Clinical Decision Support Generative Engine")
    parser.add_argument("query", nargs="?", default="ما هي معايير تشخيص مرض السكري؟", help="Clinical question (Arabic or English)")
    parser.add_argument("--k", type=int, default=3, help="Number of evidence chunks to synthesize")
    parser.add_argument("--doc", type=str, default=None, help="Filter to specific document_name")
    parser.add_argument("--lang", type=str, default="en", help="Language code ('ar' or 'en')")

    args = parser.parse_args()
    generate_clinical_recommendation(args.query, top_k=args.k, doc_filter=args.doc, lang=args.lang)

if __name__ == "__main__":
    main()
