import pandas as pd 
import utils.prompt_utils as prompt_utils
import utils.utils as utils
import generators.random_generators as random_generators
import json
from typing import Dict, Any
import json
import requests
import dotenv
from openai import OpenAI
import os
from pdfminer.high_level import extract_text
import fitz  # PyMuPDF
import re
import json
from typing import Dict, Any
import time
import httpx

# ---- Load API key from .env ----
dotenv.load_dotenv()
API_KEY = dotenv.get_key('.env', 'api_key')

# --- Init OpenAI client ---
def get_openai_client():
    return OpenAI(api_key=API_KEY)

def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    """
    Retry a function with exponential backoff.
    Catches httpx.TimeoutException and transient HTTP/network errors.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt)
            print(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s...")
            time.sleep(delay)
        except Exception as e:
            # For non-network errors, don't retry
            raise e

def generate_context_report(company_name: str) -> dict:
    """
    Generate a full context for the given company.
    """

    # Check if report exists; 
    if not os.path.exists(f"data/inputdata/reports/{company_name}.pdf"):
        print(f"The model will produce the data without the year-end report for {company_name} as context...")
        data_report = None
        #data_report = generate_year_end_report_with_web_llm(company_name=company_name)
    else:
        print(f"Loading and reading year-end report for {company_name}.")
        print(f"This usually takes ~ 5 mins...")
        data_report = generate_year_end_report_from_pdf(company_name=company_name)
        with open(f"data/inputdata/reports/generated/{company_name}_context_report.txt", "w", encoding="utf-8") as f:
            f.write(str(data_report))
    print(f"Year-end report for {company_name} is ready.")
    return data_report

def _read_text(path: str, max_chars: int = 8_000, enc: str = "utf-8") -> str:
    try:
        with open(path, "r", encoding=enc) as f:
            return f.read()[:max_chars]
    except FileNotFoundError:
        return ""


def generate_context_numbers_llm(
    company_name: str,
    model: str = "gpt-4.1",
    temp: float = 0.8  # steadier
    ) -> Dict[str, Any]:

    client = get_openai_client()

    # Load context (short to stay within budget)
    input_file = f"data/inputdata/reports/generated/{company_name}_context_report.txt"
    ctx = _read_text(input_file)

    system = f"""
    You are a precise data configurator. Return ONLY a valid **minified** JSON object and nothing else.
    Schema:
    
    "company_name": str,
    "count_employee": int,
    "count_department": int,
    "count_customer": int,
    "count_product": int,
    "count_procurement": int,
    "count_service": int,
    "count_account": int,
    "count_vendor": int,
    "estimated_product": int,
    "estimated_service": int,
    "estimated_overhead": int,
    "estimated_revenue": int
    

    Scope & scale:
    - Denmark-only operations (NOT global).
    - Treat the company as **large-size**. Use the below guidelines: 

    Ranges = 
        "count_employee":   (200, 600),
        "count_department": (10, 30),
        "count_customer":   (40, 60),
        "count_product":    (100,  200),
        "count_procurement":(80,  150),
        "count_service":    (50,  150),
        "count_vendor":     (40,  80),  
        "count_account":    (30, 80),

        Rules:
        - Prefer from the context. If the context gives global totals, **scale down** to large-size Denmark using the bands above.
        - Avoid zeros for counts and costs; integers only.
        - Output **exactly** the keys in the schema; no extra text, no comments.
        - The last *estimated* key must reflect the financials in DKK, find these in the shortened year-end report: 
        *REMEMBER*: IF A NUMBER IS DIVIDED WITH "," IT IS A DECIMAL POINT IN THE DANISH SYSTEM! I.e. 14,200 Million DKK = 14.2 Million DKK.
        {ctx}

        """.strip()

    user = f"""
    Company: {company_name}
    Task: Produce the Denmark-only configuration dict per the schema.
    Use these context excerpts to ground the numbers. If they are global, scale to mid-size Denmark using the guardrails.
    Context (may be noisy):
    \"\"\"{ctx if ctx else 'No context found. Use the mid-size Denmark guardrails.'}\"\"\"
    """.strip()

    resp = retry_with_backoff(lambda: client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temp,
        timeout=60.0,
    ))

    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
    except Exception:
        # Minimal safe fallback in mid-size bands
        data = {
            "company_name": company_name,
            "count_employee": 300,
            "count_department": 10,
            "count_customer": 250,
            "count_product": 50,
            "count_procurement": 80,
            "count_service": 40,
            "count_account": 140,
            "estimated_revenue": 1_600_000_000,
            "estimated_product": 560_000_000,   # 35%
            "estimated_service": 224_000_000,   # 14%
            "estimated_overhead": 96_000_000    # 6%
        }

    data["company_name"] = company_name
    return data


def extract_pdf_text(company_name, max_pages=None):
    pdf_path = f"data/inputdata/reports/{company_name}.pdf"
    
    # Check if the file exists
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"File not found: {pdf_path}")
    
    try:
        text_chunks = []
        with fitz.open(pdf_path) as doc:
            n_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
            for i in range(n_pages):
                page = doc[i]
                text_chunks.append(page.get_text("text"))
        return "\n".join(text_chunks)
    
    except Exception as e:
        # Fallback to pdfminer.six
        try:
            return extract_text(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

# ---------- Cleaning helpers ----------
def _clean_text(txt: str, max_chars: int = 180_000) -> str:
    # Remove excessive whitespace / hyphenation artifacts
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    # Unwrap common hyphenated line breaks (e.g., “inter-\nnational” → “international”)
    txt = re.sub(r"-\n", "", txt)
    # Normalize newlines
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    # Trim to keep prompt affordable
    return txt[:max_chars].strip()

def chunk_text(text, chunk_size=3500):
    """
    Breaks text into chunks of a given size, trying not to cut off sentences.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1  # Including space
        if current_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))  # Add remaining words as the last chunk
    return chunks

def generate_year_end_report_from_pdf(
        company_name: str,
        model: str = "gpt-4.1",
        temp: float = 1,
        dk_scope: bool = True
    ) -> str:

    pdf_path = f"data/inputdata/reports/{company_name}.pdf"
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    raw_text = extract_pdf_text(company_name)
    source_text = _clean_text(raw_text, max_chars=180_000)
    chunks = chunk_text(source_text, chunk_size=3500)

    client = get_openai_client()

    # 1) MAP — extract KPIs/claims per chunk as JSON lines
    kpi_snippets = []
    for chunk in chunks:
        scope = "Denmark-only, if present in text; otherwise general." if dk_scope else "Global or as stated."
        map_prompt = f"""
        Extract financial KPIs and claims as JSON lines from the excerpt. Prefer exact figures.
        Fields: metric, value, unit, period, note (<=25 words).
        Scope: {scope}
        Return ONLY JSON lines, one object per line. No prose.
        IF THE REPORT IS WRITTEN IN DANISH, USE THE DANISH NUMERICAL SYSTEM IN THE OUTPUT (e.g., 14,200 Million DKK = 14.2 Million DKK).
        EXCERPT:
        \"\"\"{chunk}\"\"\"
        """.strip()

        r = retry_with_backoff(lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract structured KPIs from noisy text. Output JSON only."},
                {"role": "user", "content": map_prompt}
            ],
            temperature=0,
            max_tokens=900,
            timeout=60.0,
        ))
        content = (r.choices[0].message.content or "").strip()
        if content:
            kpi_snippets.append(content)

    kpis_text = "\n".join(kpi_snippets).strip()
    if not kpis_text:
        # Fallback: if nothing extracted, reduce on a tiny summary of the first chunk
        kpis_text = '{"metric":"note","value":"","unit":"","period":"","note":"No explicit KPIs found; summarize qualitatively."}'

    # 2) REDUCE — single cohesive narrative
    scope_snippet = " (Denmark scope)" if dk_scope else ""
    reduce_prompt = f"""
    Using the JSON KPIs/claims below, write a single cohesive year-end report for {company_name}{scope_snippet}.
    - Focus on revenue, procurement costs, margins, investments, cost control, and sustainability spend.
    - Use exact numbers where available; do not invent precision.
    - 6–10 paragraphs, concise, no bullet lists.
    - Add information on services, products and procurement if available.

    KPIs/CLAIMS (JSON lines):
    {kpis_text}
    """.strip()

    r2 = retry_with_backoff(lambda: client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You craft crisp corporate year-end summaries focused on financial performance."},
            {"role": "user", "content": reduce_prompt}
        ],
        temperature=temp,
        max_tokens=1500,
        timeout=60.0,
    ))
    return (r2.choices[0].message.content or "").strip()

