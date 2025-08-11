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

# ---- Load API key from .env ----
dotenv.load_dotenv()
API_KEY = dotenv.get_key('.env', 'api_key')

# --- Init OpenAI client ---
def get_openai_client():
    return OpenAI(api_key=API_KEY)

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
        print(f"This usually takes around 5 mins...")
        data_report = generate_year_end_report_from_pdf(company_name=company_name)
        with open(f"data/inputdata/reports/generated/{company_name}_context_report.txt", "w", encoding="utf-8") as f:
            f.write(str(data_report))
    print(f"Year-end report for {company_name} is ready.")
    return data_report

import json, os, re
from typing import Dict, Any

def _read_text(path: str, max_chars: int = 8_000, enc: str = "utf-8") -> str:
    try:
        with open(path, "r", encoding=enc) as f:
            return f.read()[:max_chars]
    except FileNotFoundError:
        return ""

def _infer_size_hint(ctx: str) -> str:
    """
    Very light heuristic: look for 'billion/milliard/bn' or big 'million' counts.
    Returns 'large', 'mid', or 'sme'.
    """
    if re.search(r"\b(billion|milliard|bn)\b", ctx, re.I) or re.search(r"\b\d{3,}\s*(million|mio)\b", ctx, re.I):
        return "large"
    if re.search(r"\b\d{2,}\s*(million|mio)\b", ctx, re.I):
        return "mid"
    return "sme"

_FLOORS = {
    "sme":   dict(emp=30,  dept=6,  cust=50,  prod=15,  proc=20,  serv=12,  acct=80,  rev=100_000_000, shares=(0.35, 0.20, 0.10)),
    "mid":   dict(emp=200, dept=8,  cust=200, prod=40,  proc=60,  serv=30,  acct=120, rev=1_000_000_000, shares=(0.40, 0.18, 0.07)),
    "large": dict(emp=1000,dept=10, cust=500, prod=80,  proc=120, serv=60,  acct=180, rev=5_000_000_000, shares=(0.45, 0.15, 0.05)),
}

def _repair_numbers(data: Dict[str, Any], size_hint: str) -> Dict[str, Any]:
    f = _FLOORS[size_hint]
    # Apply minimum floors to counts
    data["count_employee"]   = max(int(data.get("count_employee", 0)), f["emp"])
    data["count_department"] = max(int(data.get("count_department", 0)), f["dept"])
    data["count_customer"]   = max(int(data.get("count_customer", 0)), f["cust"])
    data["count_product"]    = max(int(data.get("count_product", 0)), f["prod"])
    data["count_procurement"]= max(int(data.get("count_procurement", 0)), f["proc"])
    data["count_service"]    = max(int(data.get("count_service", 0)), f["serv"])
    data["count_account"]    = max(int(data.get("count_account", 0)), f["acct"])

    # Pull estimates and sanitize
    rev  = int(data.get("estimated_revenue", 0))
    prod = int(data.get("estimated_product", 0))
    serv = int(data.get("estimated_service", 0))
    ovh  = int(data.get("estimated_overhead", 0))

    # Ensure a sensible revenue floor
    rev = max(rev, f["rev"])

    # If any cost buckets are zero or negative, backfill from splits
    s_prod, s_serv, s_ovh = f["shares"]
    if prod <= 0 or serv <= 0 or ovh <= 0:
        prod = max(prod, int(rev * s_prod))
        serv = max(serv, int(rev * s_serv))
        ovh  = max(ovh,  int(rev * s_ovh))

    # If costs exceed revenue, scale them down proportionally to 85% of revenue
    total_cost = prod + serv + ovh
    if total_cost > rev:
        scale = int(rev * 0.85)
        if total_cost > 0:
            prod = int(prod * scale / total_cost)
            serv = int(serv * scale / total_cost)
            ovh  = int(ovh  * scale / total_cost)
            total_cost = prod + serv + ovh

    # If revenue is implausibly close to zero or below floors, bump
    rev = max(rev, prod + serv + ovh + int(0.10 * rev))  # keep >= costs with ~10% headroom

    data["estimated_revenue"]  = int(rev)
    data["estimated_product"]  = int(prod)
    data["estimated_service"]  = int(serv)
    data["estimated_overhead"] = int(ovh)
    return data


def generate_context_numbers_llm(
    company_name: str,
    model: str = "gpt-4.1",
    temp: float = 0.8  # steadier
) -> Dict[str, Any]:
    client = get_openai_client()

    # Load context (short to stay within budget)
    input_file = f"data/inputdata/reports/generated/{company_name}_context_report.txt"
    ctx = _read_text(input_file, max_chars=8_000)

    system = """
You are a precise data configurator. Return ONLY a valid **minified** JSON object and nothing else.
Schema:
{
  "company_name": str,
  "count_employee": int,
  "count_department": int,
  "count_customer": int,
  "count_product": int,
  "count_procurement": int,
  "count_service": int,
  "count_account": int,
  "estimated_product": int,
  "estimated_service": int,
  "estimated_overhead": int,
  "estimated_revenue": int
}

Scope & scale:
- Denmark-only operations (NOT global).
- Treat the company as **large-size**. Use the below guidelines: 

Ranges = {
    "count_employee":   (150, 600),
    "count_department": (8,   20),
    "count_customer":   (100, 200),
    "count_product":    (80,  100),
    "count_procurement":(80,  100),
    "count_service":    (40,  100),
    "count_account":    (110, 180),
    "estimated_revenue": sim 500_000-800_000 * count_employee,  
    "estimated_product": sim 30-40% * estimated_revenue,  
    "estimated_service": sim 12-18% * estimated_revenue, 
    "estimated_overhead": sim 4-8% * estimated_revenue 
}

Rules:
- Prefer from the context. If the context gives global totals, **scale down** to large-size Denmark using the bands above.
- Avoid zeros for counts and costs; integers only.
- Output **exactly** the keys in the schema; no extra text, no comments.
- The last *estimated* key must reflect the estimated financials in DKK.
""".strip()

    user = f"""
Company: {company_name}
Task: Produce the Denmark-only configuration dict per the schema.
Use these context excerpts to ground the numbers. If they are global, scale to mid-size Denmark using the guardrails.
Context (may be noisy):
\"\"\"{ctx if ctx else 'No context found. Use the mid-size Denmark guardrails.'}\"\"\"
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temp,
    )

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
        # Attempt to use PyMuPDF (fitz)
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

EXCERPT:
\"\"\"{chunk}\"\"\"
""".strip()

        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract structured KPIs from noisy text. Output JSON only."},
                {"role": "user", "content": map_prompt}
            ],
            temperature=0,
            max_tokens=900,
        )
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

    r2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You craft crisp corporate year-end summaries focused on financial performance."},
            {"role": "user", "content": reduce_prompt}
        ],
        temperature=temp,
        max_tokens=1500,
    )
    return (r2.choices[0].message.content or "").strip()


## FRAMEWORK FOR BUILDING CONTEXT USING WEB-API's: 


# ---- Search function (replace with your own search provider) ----
def search_web_impl(query: str) -> str:
    """
    Search the web for relevant URLs.
    Replace `my_search_api` with your own search logic (Bing, SerpAPI, etc.).
    Must return a JSON string with a short list of URLs.
    """
    results = my_search_api(query)  # <-- implement
    return json.dumps([r["url"] for r in results[:5]])

# ---- URL fetch function ----
def fetch_url_impl(url: str) -> str:
    """
    Fetch a URL and return extracted text.
    Handles HTML and PDFs.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "pdf" in content_type.lower():
        return extract_text_from_pdf(resp.content)[:200_000]
    return resp.text[:200_000]  # trim for token safety

# ---- PDF text extraction ----
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text

# ---- Main callable ----
def generate_year_end_report_with_web_llm(company_name: str, model: str = "gpt-5", temp: float = 1) -> str:
    client = get_openai_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search for relevant URLs given a query.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch a URL and return its raw text (HTML or extracted PDF).",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"]
                }
            }
        }
    ]

    system = (
        "You are a sharp corporate communications writer. "
        "You can request 'search_web' to find the latest Denmark year-end or annual report for the company, "
        "then call 'fetch_url' to retrieve and use its text. "
        "Summarize into 3–5 narrative paragraphs (operations highlights, commercial performance, major initiatives, outlook). "
        "Use only qualitative wording (e.g. 'strong growth', 'soft demand'). No KPIs, no tables, no bullet points."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Draft a Denmark-scope year-end report for {company_name}."}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temp
    )
    msg = resp.choices[0].message

    # Loop until no more tool calls
    while getattr(msg, "tool_calls", None):
        for call in msg.tool_calls:
            func = call.function.name
            args = json.loads(call.function.arguments)
            if func == "search_web":
                content = search_web_impl(**args)
            elif func == "fetch_url":
                content = fetch_url_impl(**args)
            else:
                content = ""
            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": func,
                "content": content
            })
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temp
        )
        msg = resp.choices[0].message

    return msg.content.strip()
