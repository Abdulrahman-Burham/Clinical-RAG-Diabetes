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

from retrieve import query_clinical_rag, load_index, retrieve

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
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=api_base)

            if is_ar:
                system_prompt = (
                    "أنت طبيب استشاري خبير متخصص في الدلائل الإرشادية لمرض السكري والكرومات والجزء الهرموني.\n"
                    "القاعدة الذهبية: لا تجب على أي سؤال بدون دعم كامل بالاستشهادات ودلائل من النصوص المرفقة.\n"
                    "يجب تضمين اسم المستند ورقم الصفحة لكل توصية سريرية تقوم بصياغتها.\n"
                    "إذا كان السؤال خارج نطاق المراجع المرفقة، اذكر بوضوح أن المراجع الحالية لا تحتوي على الإجابة."
                )
                user_prompt = f"الاستفسار السريري: {query_text}\n\nالنصوص والدلائل المتاحة:\n{full_context}\n\nيرجى تقديم توصية سريرية دقيقة ومسببة مدعمة بالمراجع والمستندات."
            else:
                system_prompt = (
                    "You are an expert Clinical Decision Support Assistant specializing in Diabetes & Endocrinology Guidelines.\n"
                    "THE GOLDEN RULE: 'No claim without a citation.' Every recommendation must include explicit inline citations [Doc Name | Section | Page].\n"
                    "Only make assertions that are supported by the provided clinical context passages. If the context does not contain the answer, explicitly state that."
                )
                user_prompt = f"Clinical Question: {query_text}\n\nAvailable Evidence Passages:\n{full_context}\n\nPlease provide a clear, citable clinical recommendation based strictly on the evidence above."

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )

            recommendation = response.choices[0].message.content.strip()
            print("\n" + "=" * 80)
            print("                 SYNTHESIZED CLINICAL RECOMMENDATION                ")
            print("=" * 80)
            print(recommendation)
            print("=" * 80 + "\n")
            return recommendation
        except Exception as e:
            print(f"\n[LLM API Error] Falling back to structured citable summary: {e}")

    # Offline / Fallback Citable Summary
    print("\n[Fallback Mode] Generating evidence-based citable summary directly from retrieved passages:")
    fallback_output = [
        f"Clinical Question: {query_text}\n",
        "Key Retrived Clinical Evidence & Guidelines:"
    ]
    for citation, block in zip(citations_list, context_blocks):
        fallback_output.append(f"\n• According to [{citation}]:")
        fallback_output.append(f"  \"{block.splitlines()[-1][:250]}...\"")

    final_fallback = "\n".join(fallback_output)
    print(final_fallback)
    return final_fallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Decision Support Generative RAG Engine")
    parser.add_argument("query", type=str, help="Clinical question or scenario")
    parser.add_argument("--k", type=int, default=3, help="Number of evidence chunks to retrieve")
    parser.add_argument("--doc", type=str, default=None, help="Filter search to specific guideline document")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "ar"], help="Language output (en/ar)")

    args = parser.parse_args()
    generate_clinical_recommendation(args.query, top_k=args.k, doc_filter=args.doc, lang=args.lang)
